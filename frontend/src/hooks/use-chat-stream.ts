import { useCallback, useEffect, useReducer, useRef } from "react"
import { streamChat } from "@/api/client"
import { chatReducer, isTurnRunning } from "@/lib/chat-turns"

let turnsCreated = 0

/** crypto.randomUUID needs a secure context, which a plain-HTTP host is not. */
function createTurnId(): string {
	turnsCreated += 1
	return globalThis.crypto?.randomUUID?.() ?? `turn-${turnsCreated}`
}

export function useChatStream() {
	const [turns, dispatch] = useReducer(chatReducer, [])
	const abort = useRef<AbortController | null>(null)

	const ask = useCallback(async (question: string) => {
		abort.current?.abort()
		const controller = new AbortController()
		abort.current = controller
		dispatch({ type: "ask", id: createTurnId(), question })
		try {
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

	const current = turns.at(-1)
	return {
		turns,
		ask,
		stop,
		isBusy: current !== undefined && isTurnRunning(current),
	}
}
