import { expect, test } from 'vitest'
import type { ChatSource, ChatTurn } from './use-chat-stream'
import { chatReducer, readChatStream } from './use-chat-stream'

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

function asked(question = 'q'): ChatTurn[] {
	return chatReducer([], { type: 'ask', id: 'turn-1', question })
}

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
	const turns = actions.reduce(chatReducer, asked())
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
	expect(actions.reduce(chatReducer, asked())[0].answer).toBe('split')
})

test('an error frame fails the turn with its message', async () => {
	const actions: Parameters<typeof chatReducer>[1][] = []
	await readChatStream(
		frames(
			'event: error\ndata: {"error":"llm_error","message":"chat call failed","request_id":"r1"}\n\n',
		),
		(action) => actions.push(action),
	)
	const turns = actions.reduce(chatReducer, asked())
	expect(turns[0].status).toBe('failed')
	expect(turns[0].error).toBe('chat call failed')
})

test('settling never overwrites a failed turn', () => {
	const failed = chatReducer(asked(), {
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
	const turns = actions.reduce(chatReducer, asked())
	expect(turns[0].sources).toEqual([])
	expect(turns[0].answer).toBe('I cannot answer that.')
})
