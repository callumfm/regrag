import { describe, expect, it } from "vitest"
import type { ChatStep } from "@/api/types"
import { formatDuration, stepLabel } from "./chat-steps"

function step(
	name: ChatStep["step"],
	ms = 0,
	status: ChatStep["status"] = "completed",
): ChatStep {
	return { step: name, ms, status }
}

describe("stepLabel", () => {
	it("names a finished step in the past tense", () => {
		expect(stepLabel(step("tool_follow_reference"))).toBe(
			"Followed a reference",
		)
	})

	it("names a running step in the present tense", () => {
		expect(stepLabel(step("retrieve", 0, "running"))).toBe(
			"Searching the corpus",
		)
	})

	it("names a step the schema does not know by its own name", () => {
		const unknown = "rerank" as ChatStep["step"]
		expect(stepLabel(step(unknown))).toBe("rerank")
	})

	it("names the refuse step by what it found", () => {
		expect(stepLabel(step("tool_refuse", 0, "running"))).toBe(
			"Finding nothing that bears on the question",
		)
		expect(stepLabel(step("tool_refuse"))).toBe(
			"Found nothing that bears on the question",
		)
	})

	it("names the rewrite step by what it does", () => {
		expect(stepLabel(step("rewrite", 0, "running"))).toBe(
			"Rewriting the question",
		)
		expect(stepLabel(step("rewrite"))).toBe("Rewrote the question")
	})
})

describe("formatDuration", () => {
	it("reports a short run in milliseconds", () => {
		expect(formatDuration([step("retrieve", 120), step("assess", 300)])).toBe(
			"420ms",
		)
	})

	it("reports a long run in seconds", () => {
		expect(formatDuration([step("retrieve", 1200), step("assess", 900)])).toBe(
			"2.1s",
		)
	})
})
