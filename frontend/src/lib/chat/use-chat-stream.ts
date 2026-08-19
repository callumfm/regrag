import { createParser } from 'eventsource-parser'
import { useCallback, useEffect, useReducer, useRef } from 'react'
import type { components } from '@/api/schema'
import { API_URL } from '@/lib/api'

export type ChatSource = components['schemas']['ChatSource']

export type ChatTurn = {
	id: string
	question: string
	answer: string
	sources: ChatSource[]
	status: 'pending' | 'streaming' | 'settled' | 'failed'
	error: string | null
}

type ChatAction =
	| { type: 'ask'; id: string; question: string }
	| { type: 'sources'; sources: ChatSource[] }
	| { type: 'token'; text: string }
	| { type: 'settle' }
	| { type: 'fail'; message: string }

function applyToTurn(turn: ChatTurn, action: ChatAction): ChatTurn {
	switch (action.type) {
		case 'sources':
			return { ...turn, sources: action.sources, status: 'streaming' }
		case 'token':
			return { ...turn, answer: turn.answer + action.text, status: 'streaming' }
		case 'settle':
			return turn.status === 'failed' ? turn : { ...turn, status: 'settled' }
		case 'fail':
			return { ...turn, status: 'failed', error: action.message }
		default:
			return turn
	}
}

export function chatReducer(turns: ChatTurn[], action: ChatAction): ChatTurn[] {
	if (action.type === 'ask') {
		return [
			...turns,
			{
				id: action.id,
				question: action.question,
				answer: '',
				sources: [],
				status: 'pending',
				error: null,
			},
		]
	}
	const current = turns.at(-1)
	if (current === undefined) return turns
	return [...turns.slice(0, -1), applyToTurn(current, action)]
}

function toAction(name: string | undefined, data: string): ChatAction | null {
	switch (name) {
		case 'sources':
			return { type: 'sources', sources: JSON.parse(data) }
		case 'token':
			return { type: 'token', text: JSON.parse(data).text }
		case 'done':
			return { type: 'settle' }
		case 'error':
			return { type: 'fail', message: JSON.parse(data).message }
		default:
			return null
	}
}

async function readErrorMessage(response: Response): Promise<string> {
	const fallback = `Chat request failed: ${response.status}`
	try {
		const body = await response.json()
		return typeof body?.message === 'string' && body.message !== ''
			? body.message
			: fallback
	} catch {
		return fallback
	}
}

export async function readChatStream(
	body: ReadableStream<Uint8Array>,
	dispatch: (action: ChatAction) => void,
): Promise<void> {
	const decoder = new TextDecoder()
	const parser = createParser({
		onEvent(event) {
			const action = toAction(event.event, event.data)
			if (action !== null) dispatch(action)
		},
	})
	const reader = body.getReader()
	try {
		while (true) {
			const { done, value } = await reader.read()
			if (done) break
			parser.feed(decoder.decode(value, { stream: true }))
		}
	} catch (error) {
		reader.cancel().catch(() => undefined)
		throw error
	} finally {
		reader.releaseLock()
	}
}

export function useChatStream() {
	const [turns, dispatch] = useReducer(chatReducer, [])
	const abort = useRef<AbortController | null>(null)

	const ask = useCallback(async (question: string) => {
		abort.current?.abort()
		const controller = new AbortController()
		abort.current = controller
		try {
			dispatch({ type: 'ask', id: crypto.randomUUID(), question })
			const response = await fetch(`${API_URL}/chat`, {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ question }),
				signal: controller.signal,
			})
			if (!response.ok) {
				throw new Error(await readErrorMessage(response))
			}
			if (response.body === null) {
				throw new Error('Chat response had no body to stream')
			}
			await readChatStream(response.body, dispatch)
			dispatch({ type: 'settle' })
		} catch (error) {
			if (controller.signal.aborted) return
			dispatch({
				type: 'fail',
				message: error instanceof Error ? error.message : 'Chat request failed',
			})
		}
	}, [])

	const stop = useCallback(() => {
		abort.current?.abort()
		dispatch({ type: 'settle' })
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
		isBusy: status === 'pending' || status === 'streaming',
	}
}
