import type { ChatSource, ChatStep, ChatStreamEvent } from "@/api/types"

export type ChatTurn = {
	id: string
	question: string
	answer: string
	sources: ChatSource[]
	steps: ChatStep[]
	status: "pending" | "streaming" | "settled" | "failed"
	error: string | null
}

export type ChatAction =
	| { type: "ask"; id: string; question: string }
	| { type: "settle" }
	| { type: "fail"; message: string }
	| ChatStreamEvent

function applyToTurn(turn: ChatTurn, action: ChatAction): ChatTurn {
	if ("event" in action) {
		switch (action.event) {
			case "sources":
				return { ...turn, sources: action.data, status: "streaming" }
			case "step":
				return {
					...turn,
					steps: [...turn.steps, action.data],
					status: "streaming",
				}
			case "text":
				return {
					...turn,
					answer: turn.answer + action.data,
					status: "streaming",
				}
			case "done":
				return turn.status === "failed" ? turn : { ...turn, status: "settled" }
			case "error":
				return { ...turn, status: "failed", error: action.data.message }
		}
	}
	switch (action.type) {
		case "settle":
			return turn.status === "failed" ? turn : { ...turn, status: "settled" }
		case "fail":
			return { ...turn, status: "failed", error: action.message }
		default:
			return turn
	}
}

export function chatReducer(turns: ChatTurn[], action: ChatAction): ChatTurn[] {
	if ("type" in action && action.type === "ask") {
		return [
			...turns,
			{
				id: action.id,
				question: action.question,
				answer: "",
				sources: [],
				steps: [],
				status: "pending",
				error: null,
			},
		]
	}
	const current = turns.at(-1)
	if (current === undefined) return turns
	return [...turns.slice(0, -1), applyToTurn(current, action)]
}
