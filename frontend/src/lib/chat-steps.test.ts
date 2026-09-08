import { describe, expect, it } from "vitest"
import type { ChatStep } from "@/api/types"
import { formatDuration, stepLabel, trailSteps } from "./chat-steps"

function step(name: string, ms = 0): ChatStep {
	return { step: name as ChatStep["step"], ms }
}

describe("trailSteps", () => {
	it("keeps the work done before the answer", () => {
		const kept = trailSteps([
			step("retrieve"),
			step("assess"),
			step("tool_search"),
		])

		expect(kept.map((s) => s.step)).toEqual([
			"retrieve",
			"assess",
			"tool_search",
		])
	})

	it("drops the steps that are the answer", () => {
		expect(trailSteps([step("retrieve"), step("synthesize")])).toHaveLength(1)
		expect(trailSteps([step("retrieve"), step("refuse")])).toHaveLength(1)
	})
})

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
