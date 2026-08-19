import { act, renderHook } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import type { ChatSource, ChatTurn } from './use-chat-stream'
import { chatReducer, readChatStream, useChatStream } from './use-chat-stream'

const SOURCE: ChatSource = {
	marker: 1,
	chunk_id: 1,
	celex: '32023R1805',
	citation: 'Article 4(1)',
	title: 'Greenhouse gas intensity limit',
	text: 'The greenhouse gas intensity of the energy used on board.',
}

function frames(...lines: string[]): ReadableStream<Uint8Array> {
	const encoder = new TextEncoder()
	return new ReadableStream({
		start(controller) {
			for (const line of lines) controller.enqueue(encoder.encode(line))
			controller.close()
		},
	})
}

function askedTurns(question = 'q'): ChatTurn[] {
	return chatReducer([], { type: 'ask', id: 'turn-1', question })
}

afterEach(() => {
	vi.unstubAllGlobals()
})

test('records sources, then tokens, then settles', async () => {
	const actions: Parameters<typeof chatReducer>[1][] = []
	await readChatStream(
		frames(
			`event: sources\ndata: ${JSON.stringify([SOURCE])}\n\n`,
			'event: token\ndata: {"text":"Ships "}\n\n',
			'event: token\ndata: {"text":"comply [1]."}\n\n',
			'event: done\ndata: {}\n\n',
		),
		(action) => actions.push(action),
	)
	const turns = actions.reduce(chatReducer, askedTurns())
	expect(turns[0].answer).toBe('Ships comply [1].')
	expect(turns[0].sources).toEqual([SOURCE])
	expect(turns[0].status).toBe('settled')
})

test('reassembles a frame split across network chunks', async () => {
	const actions: Parameters<typeof chatReducer>[1][] = []
	await readChatStream(
		frames('event: token\ndata: {"te', 'xt":"split"}\n\n'),
		(action) => actions.push(action),
	)
	expect(actions.reduce(chatReducer, askedTurns())[0].answer).toBe('split')
})

test('an error frame fails the turn with its message', async () => {
	const actions: Parameters<typeof chatReducer>[1][] = []
	await readChatStream(
		frames(
			'event: error\ndata: {"error":"llm_error","message":"chat call failed","request_id":"r1"}\n\n',
		),
		(action) => actions.push(action),
	)
	const turns = actions.reduce(chatReducer, askedTurns())
	expect(turns[0].status).toBe('failed')
	expect(turns[0].error).toBe('chat call failed')
})

test('settling never overwrites a failed turn', () => {
	const failed = chatReducer(askedTurns(), {
		type: 'fail',
		message: 'chat call failed',
	})
	expect(chatReducer(failed, { type: 'settle' })[0].status).toBe('failed')
})

test('a refusal arrives as an empty sources list and one token', async () => {
	const actions: Parameters<typeof chatReducer>[1][] = []
	await readChatStream(
		frames(
			'event: sources\ndata: []\n\n',
			'event: token\ndata: {"text":"I cannot answer that."}\n\n',
			'event: done\ndata: {}\n\n',
		),
		(action) => actions.push(action),
	)
	const turns = actions.reduce(chatReducer, askedTurns())
	expect(turns[0].sources).toEqual([])
	expect(turns[0].answer).toBe('I cannot answer that.')
})

test('unmounting the hook aborts the in-flight request', async () => {
	let capturedSignal: AbortSignal | undefined
	vi.stubGlobal(
		'fetch',
		vi.fn((_url: string, init?: RequestInit) => {
			capturedSignal = init?.signal ?? undefined
			return new Promise(() => {})
		}),
	)

	const { result, unmount } = renderHook(() => useChatStream())
	act(() => {
		result.current.ask('What is FuelEU?')
	})

	expect(capturedSignal?.aborted).toBe(false)
	unmount()
	expect(capturedSignal?.aborted).toBe(true)
})

test('a non-ok response surfaces the backend message instead of the status', async () => {
	vi.stubGlobal(
		'fetch',
		vi.fn().mockResolvedValue({
			ok: false,
			status: 429,
			json: async () => ({
				error: 'rate_limited',
				message: 'Too many requests, try again shortly.',
				request_id: 'r1',
			}),
		}),
	)

	const { result } = renderHook(() => useChatStream())
	await act(async () => {
		await result.current.ask('What is FuelEU?')
	})

	expect(result.current.turns[0].status).toBe('failed')
	expect(result.current.turns[0].error).toBe(
		'Too many requests, try again shortly.',
	)
})

test('an ok response with no body fails without claiming the status failed', async () => {
	vi.stubGlobal(
		'fetch',
		vi.fn().mockResolvedValue({ ok: true, status: 200, body: null }),
	)

	const { result } = renderHook(() => useChatStream())
	await act(async () => {
		await result.current.ask('What is FuelEU?')
	})

	expect(result.current.turns[0].status).toBe('failed')
	expect(result.current.turns[0].error).not.toMatch(/200/)
})
