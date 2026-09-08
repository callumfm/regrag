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

/** How long the run spent on the steps it finished, as the collapsed trail reports it. */
export function formatDuration(steps: ChatStep[]): string {
	const ms = steps.reduce((total, step) => total + step.ms, 0)
	return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`
}
