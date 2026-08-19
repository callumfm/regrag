import { useCallback, useEffect, useReducer, useRef } from "react"
import { streamChat } from "@/api/client"
import type { ChatSource, ChatStreamEvent } from "@/api/types"

export type ChatTurn = {
	id: string
	question: string
	answer: string
	sources: ChatSource[]
	status: "pending" | "streaming" | "settled" | "failed"
	error: string | null
}

type ChatAction =
	| { type: "ask"; id: string; question: string }
	| { type: "settle" }
	| { type: "fail"; message: string }
	| ChatStreamEvent

function applyToTurn(turn: ChatTurn, action: ChatAction): ChatTurn {
	if ("event" in action) {
		switch (action.event) {
			case "sources":
				return { ...turn, sources: action.data, status: "streaming" }
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
				status: "pending",
				error: null,
			},
		]
	}
	const current = turns.at(-1)
	if (current === undefined) return turns
	return [...turns.slice(0, -1), applyToTurn(current, action)]
}

export function useChatStream() {
	const [turns, dispatch] = useReducer(chatReducer, [])
	const abort = useRef<AbortController | null>(null)

	const ask = useCallback(async (question: string) => {
		abort.current?.abort()
		const controller = new AbortController()
		abort.current = controller
		try {
			dispatch({ type: "ask", id: crypto.randomUUID(), question })
			for await (const event of streamChat({ question }, controller.signal)) {
				dispatch(event)
			}
			dispatch({ type: "settle" })
		} catch (error) {
			if (controller.signal.aborted) return
			dispatch({
				type: "fail",
				message: error instanceof Error ? error.message : "Chat request failed",
			})
		}
	}, [])

	const stop = useCallback(() => {
		abort.current?.abort()
		dispatch({ type: "settle" })
	}, [])

	useEffect(() => {
		return () => {
			abort.current?.abort()
		}
	}, [])

	const status = turns.at(-1)?.status
	return {
		turns,
		ask,
		stop,
		isBusy: status === "pending" || status === "streaming",
	}
}
