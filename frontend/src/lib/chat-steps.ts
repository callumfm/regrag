import type { ChatStep } from "@/api/types"

const LABELS: Record<string, string> = {
	retrieve: "Searched the corpus",
	assess: "Reviewed the evidence",
	tool_search: "Searched again",
	tool_follow_reference: "Followed a reference",
	tool_unknown: "Asked for a tool it does not have",
}

const ANSWER_STEPS = new Set(["synthesize", "refuse"])

/** The steps worth showing: the work before the answer, which the answer itself is not. */
export function trailSteps(steps: ChatStep[]): ChatStep[] {
	return steps.filter((step) => !ANSWER_STEPS.has(step.step))
}

/** What a step is called in the trail; one with no label of its own goes by its name. */
export function stepLabel(step: ChatStep): string {
	const named = LABELS[step.step]
	if (named !== undefined) return named
	const words = step.step.replaceAll("_", " ")
	return words.charAt(0).toUpperCase() + words.slice(1)
}

/** How long the run spent on the steps, as the collapsed trail reports it. */
export function formatDuration(steps: ChatStep[]): string {
	const ms = steps.reduce((total, step) => total + step.ms, 0)
	return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`
}
