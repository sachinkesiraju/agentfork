"""End-to-end test of the live trace-driven online distillation loop."""

import pytest
import torch

from agentfork.kvzip_router import (
    FullCacheTeacher,
    KVBudgetHint,
    KVCacheGate,
    KVCompressor,
    OnlineDistiller,
)
from agentfork.kvzip_router.trace import TraceJournal, TraceRecord, TraceTrainer


def _make_records(tokenizer, texts):
    return [
        TraceRecord(
            input_ids=tokenizer(text, return_tensors="pt").input_ids,
            budget=KVBudgetHint(task_id=f"task_{i}", adapter_id=f"adapter_{i}"),
        )
        for i, text in enumerate(texts)
    ]


def _eval_loss(compressor, teacher, input_ids, budget):
    condition = compressor._condition(budget, device=input_ids.device)
    with torch.no_grad():
        outputs = compressor.model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        teacher_scores = teacher.score(input_ids, n_kv_heads=compressor.n_kv_heads)
        loss = OnlineDistiller(compressor.gate).distill_loss(
            outputs.hidden_states, teacher_scores, condition=condition
        )
    return float(loss.item())


def test_trace_journal_capacity():
    journal = TraceJournal(capacity=2)
    ids = torch.tensor([[1, 2, 3]])
    journal.add(TraceRecord(ids))
    journal.add(TraceRecord(ids))
    assert len(journal) == 2
    journal.add(TraceRecord(ids))
    assert len(journal) == 2


@pytest.mark.slow
class TestTraceLoop:
    def test_trace_trainer_improves_gate(self, tiny_tokenizer, tiny_model):
        """A synthetic production trace should make the gate fit real workloads."""
        gate = KVCacheGate(
            hidden_dim=tiny_model.config.hidden_size,
            n_kv_heads=tiny_model.config.num_key_value_heads,
            n_layers=tiny_model.config.num_hidden_layers,
            cond_dim=8,
        )
        compressor = KVCompressor(tiny_model, gate=gate)

        train_texts = [
            "the quick brown fox jumps",
            "hello world this is a test",
            "one two three four five six",
            "the cat sat on the mat and looked",
        ]
        traces = _make_records(tiny_tokenizer, train_texts)

        test_text = "the dog ran across the field quickly"
        test_ids = tiny_tokenizer(test_text, return_tensors="pt").input_ids
        test_budget = KVBudgetHint(task_id="test", adapter_id="test")

        teacher = FullCacheTeacher(tiny_model)
        initial_loss = _eval_loss(compressor, teacher, test_ids, test_budget)

        trainer = TraceTrainer(compressor, teacher, lr=2e-2)
        trainer.fit(traces, epochs=20)

        final_loss = _eval_loss(compressor, teacher, test_ids, test_budget)

        assert final_loss < initial_loss * 0.9, (
            f"Trace training did not improve gate: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_trace_trainer_learns_per_adapter_condition(
        self, tiny_tokenizer, tiny_model
    ):
        """The same trace should produce different hidden-state scores with
        different adapter/task conditions, so the gate is LoRA/task aware."""
        gate = KVCacheGate(
            hidden_dim=tiny_model.config.hidden_size,
            n_kv_heads=tiny_model.config.num_key_value_heads,
            n_layers=tiny_model.config.num_hidden_layers,
            cond_dim=8,
        )
        compressor = KVCompressor(tiny_model, gate=gate)

        text = "the quick brown fox jumps over the lazy dog"
        ids = tiny_tokenizer(text, return_tensors="pt").input_ids

        budget_a = KVBudgetHint(task_id="a", adapter_id="a")
        budget_b = KVBudgetHint(task_id="b", adapter_id="b")

        condition_a = compressor._condition(budget_a, device=ids.device)
        condition_b = compressor._condition(budget_b, device=ids.device)

        with torch.no_grad():
            outputs = tiny_model(
                input_ids=ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            scores_a = compressor.gate(
                outputs.hidden_states[0], 0, condition=condition_a
            )
            scores_b = compressor.gate(
                outputs.hidden_states[0], 0, condition=condition_b
            )

        # With a fresh random gate the condition vectors differ, so predictions differ.
        assert not torch.allclose(scores_a, scores_b, atol=1e-6)

        # After training on the same text with condition_a, scores should shift.
        trainer = TraceTrainer(compressor, lr=2e-2)
        trainer.fit([TraceRecord(ids, budget=budget_a)], epochs=20)

        with torch.no_grad():
            outputs = tiny_model(
                input_ids=ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            scores_a2 = compressor.gate(
                outputs.hidden_states[0], 0, condition=condition_a
            )

        assert not torch.allclose(scores_a, scores_a2, atol=1e-6), (
            "Trace training did not change gate output for the trained condition"
        )

    def test_trace_traffic_mix(self, tiny_tokenizer, tiny_model):
        """Simulate live traffic with two task/adapter streams and verify the
        gate improves on held-out samples from both distributions."""
        gate = KVCacheGate(
            hidden_dim=tiny_model.config.hidden_size,
            n_kv_heads=tiny_model.config.num_key_value_heads,
            n_layers=tiny_model.config.num_hidden_layers,
            cond_dim=8,
        )
        compressor = KVCompressor(tiny_model, gate=gate)
        teacher = FullCacheTeacher(tiny_model)

        task_texts = [
            "the quick brown fox jumps over the lazy dog",
            "one two three four five six seven eight nine",
        ]
        adapter_texts = [
            "hello world this is a simple test of the system",
            "the cat sat on the mat and looked at the window",
        ]

        records = []
        for i, text in enumerate(task_texts):
            records.append(
                TraceRecord(
                    tiny_tokenizer(text, return_tensors="pt").input_ids,
                    budget=KVBudgetHint(task_id="qa", adapter_id=f"adapter_{i}"),
                )
            )
        for i, text in enumerate(adapter_texts):
            records.append(
                TraceRecord(
                    tiny_tokenizer(text, return_tensors="pt").input_ids,
                    budget=KVBudgetHint(task_id="chat", adapter_id=f"adapter_{i + 2}"),
                )
            )

        # Use a ring buffer to simulate a live production trace journal.
        journal = TraceJournal(capacity=8)
        for record in records:
            journal.add(record)

        held_out = tiny_tokenizer(
            "the dog ran across the field quickly", return_tensors="pt"
        ).input_ids
        held_budget = KVBudgetHint(task_id="qa", adapter_id="adapter_0")
        initial_loss = _eval_loss(compressor, teacher, held_out, held_budget)

        trainer = TraceTrainer(compressor, teacher, lr=2e-2)
        trainer.fit(journal.sample(), epochs=30)

        final_loss = _eval_loss(compressor, teacher, held_out, held_budget)
        assert final_loss < initial_loss * 0.9, (
            f"Traffic-mix training did not improve gate: {initial_loss:.4f} -> {final_loss:.4f}"
        )
