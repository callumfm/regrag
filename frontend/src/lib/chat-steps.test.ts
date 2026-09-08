import { describe, expect, it } from "vitest"
import type { ChatStep } from "@/api/types"
import { formatDuration, orbState, runningStep, stepLabel } from "./chat-steps"

function step(name: string, ms = 0, status = "completed"): ChatStep {
	return {
		step: name as ChatStep["step"],
		ms,
		status: status as ChatStep["status"],
	}
}

describe("stepLabel", () => {
	it("names a known step in English", () => {
		expect(stepLabel(step("tool_follow_reference"))).toBe(
			"Followed a reference",
		)
	})

	it("falls back to the step's own name when the graph grows one", () => {
		expect(stepLabel(step("decompose"))).toBe("Decompose")
		expect(stepLabel(step("tool_widen_search"))).toBe("Tool widen search")
	})
})

describe("runningStep", () => {
	it("finds the step the run is on", () => {
		const found = runningStep([
			step("retrieve", 12),
			step("assess", 0, "running"),
		])

		expect(found?.step).toBe("assess")
	})

	it("finds nothing once every step has settled", () => {
		expect(runningStep([step("retrieve", 12)])).toBeUndefined()
	})

	it("names the answer as a step like any other", () => {
		const steps = [step("retrieve", 12), step("synthesize", 0, "running")]

		expect(stepLabel(runningStep(steps) as ChatStep)).toBe("Writing the answer")
		expect(stepLabel(step("synthesize", 1200))).toBe("Wrote the answer")
	})
})

describe("orbState", () => {
	it("animates retrieval and tool calls as searching", () => {
		expect(orbState(step("retrieve"))).toBe("searching")
		expect(orbState(step("tool_follow_reference"))).toBe("searching")
	})

	it("falls back to working for a step it does not know", () => {
		expect(orbState(step("decompose"))).toBe("working")
		expect(orbState(undefined)).toBe("working")
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
