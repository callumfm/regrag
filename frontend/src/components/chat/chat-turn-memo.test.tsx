import { fireEvent, render, screen } from '@testing-library/react'
import { memo, type ReactNode } from 'react'
import { afterEach, expect, test, vi } from 'vitest'
import type { ChatSource, ChatTurn as Turn } from '@/lib/chat/use-chat-stream'
import { Chat } from './chat'

type ChatTurnProps = {
	turn: Turn
	onOpenMarker: (marker: number) => void
	onRetry: () => void
}

const chatTurnRenders = vi.hoisted(() => vi.fn<(turnId: string) => void>())

vi.mock('./chat-turn', async (importOriginal) => {
	const actual = await importOriginal<typeof import('./chat-turn')>()
	const rawChatTurn = (
		actual.ChatTurn as unknown as {
			type: (props: ChatTurnProps) => ReactNode
		}
	).type
	return {
		...actual,
		ChatTurn: memo((props: ChatTurnProps) => {
			chatTurnRenders(props.turn.id)
			return rawChatTurn(props)
		}),
	}
})

const TURN_1_SOURCE: ChatSource = {
	marker: 1,
	chunk_id: 1,
	celex: '32023R1805',
	citation: 'Article 4(1)',
	title: 'Greenhouse gas intensity limit',
	text: 'The greenhouse gas intensity of the energy used on board.',
}

const TURN_2_SOURCE: ChatSource = {
	marker: 1,
	chunk_id: 9,
	celex: '32023R1806',
	citation: 'Article 9(2)',
	title: 'Monitoring plan',
	text: 'Companies shall submit a monitoring plan by the compliance date.',
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

function streamingSseResponse(source: ChatSource, tokens: string[]) {
	return {
		ok: true,
		body: frames(
			`event: sources\ndata: ${JSON.stringify([source])}\n\n`,
			...tokens.map(
				(token) => `event: token\ndata: ${JSON.stringify({ text: token })}\n\n`,
			),
			'event: done\ndata: {}\n\n',
		),
	}
}

function ask(question: string) {
	fireEvent.change(screen.getByPlaceholderText('Ask about EU regulation…'), {
		target: { value: question },
	})
	fireEvent.click(screen.getByRole('button', { name: 'Send' }))
}

afterEach(() => {
	vi.unstubAllGlobals()
	chatTurnRenders.mockClear()
})

test('a settled turn does not re-render while a later turn streams tokens', async () => {
	vi.stubGlobal(
		'fetch',
		vi
			.fn()
			.mockResolvedValueOnce(
				streamingSseResponse(TURN_1_SOURCE, ['First ', 'answer', ' [1].']),
			)
			.mockResolvedValueOnce(
				streamingSseResponse(TURN_2_SOURCE, ['Second ', 'answer', ' [1].']),
			),
	)

	render(<Chat />)

	ask('What is FuelEU?')
	await screen.findByText(/First answer/)

	const turn1Id = chatTurnRenders.mock.calls.at(-1)?.[0]
	expect(turn1Id).toBeDefined()
	const rendersBeforeTurn2 = chatTurnRenders.mock.calls.filter(
		([id]) => id === turn1Id,
	).length

	ask('What about monitoring?')
	await screen.findByText(/Second answer/)

	const rendersAfterTurn2 = chatTurnRenders.mock.calls.filter(
		([id]) => id === turn1Id,
	).length

	expect(rendersAfterTurn2).toBe(rendersBeforeTurn2)
})
