import type { OrbState } from "thinking-orbs"
import type { ChatStep } from "@/api/types"

const LABELS: Record<string, { running: string; done: string }> = {
	retrieve: { running: "Searching the corpus", done: "Searched the corpus" },
	assess: { running: "Reviewing the evidence", done: "Reviewed the evidence" },
	tool_search: { running: "Searching again", done: "Searched again" },
	tool_follow_reference: {
		running: "Following a reference",
		done: "Followed a reference",
	},
	tool_unknown: {
		running: "Asking for a tool it does not have",
		done: "Asked for a tool it does not have",
	},
	synthesize: { running: "Writing the answer", done: "Wrote the answer" },
	refuse: { running: "Declining to answer", done: "Declined to answer" },
}

const ORB_STATES: Record<string, OrbState> = {
	retrieve: "searching",
	tool_search: "searching",
	tool_follow_reference: "searching",
	assess: "solving",
	synthesize: "composing",
}

/** What a step is called in the trail, in the tense its status calls for; one with no label
 * of its own goes by its name. */
export function stepLabel(step: ChatStep): string {
	const named = LABELS[step.step]
	if (named !== undefined) {
		return step.status === "running" ? named.running : named.done
	}
	const words = step.step.replaceAll("_", " ")
	return words.charAt(0).toUpperCase() + words.slice(1)
}

/** The step the run is on, which the header names and the orb animates. */
export function runningStep(steps: ChatStep[]): ChatStep | undefined {
	return steps.findLast((step) => step.status === "running")
}

/** Which orb the running step animates; anything unmapped just works. */
export function orbState(step: ChatStep | undefined): OrbState {
	return (step && ORB_STATES[step.step]) ?? "working"
}

/** How long the run spent on the steps it finished, as the collapsed trail reports it. */
export function formatDuration(steps: ChatStep[]): string {
	const ms = steps.reduce((total, step) => total + step.ms, 0)
	return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`
}
