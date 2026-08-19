import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import type { ChatSource } from '@/lib/chat/use-chat-stream'
import { Chat } from './chat'

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

function sseResponse(source: ChatSource, answer: string) {
	return {
		ok: true,
		body: frames(
			`event: sources\ndata: ${JSON.stringify([source])}\n\n`,
			`event: token\ndata: ${JSON.stringify({ text: answer })}\n\n`,
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
})

test('a citation chip in an earlier turn opens that turn’s own source, not the latest turn’s', async () => {
	vi.stubGlobal(
		'fetch',
		vi
			.fn()
			.mockResolvedValueOnce(sseResponse(TURN_1_SOURCE, 'First answer [1].'))
			.mockResolvedValueOnce(sseResponse(TURN_2_SOURCE, 'Second answer [1].')),
	)

	render(<Chat />)

	ask('What is FuelEU?')
	await screen.findByText(/First answer/)

	ask('What about monitoring?')
	await screen.findByText(/Second answer/)

	const chips = screen.getAllByRole('button', { name: 'Open source 1' })
	fireEvent.click(chips[0])

	expect(await screen.findByText(TURN_1_SOURCE.text)).toBeInTheDocument()
	expect(screen.queryByText(TURN_2_SOURCE.text)).toBeNull()
})

test('retrying a failed turn re-asks its question and can succeed', async () => {
	vi.stubGlobal(
		'fetch',
		vi
			.fn()
			.mockResolvedValueOnce({
				ok: false,
				status: 500,
				json: async () => ({
					error: 'llm_error',
					message: 'The model is unavailable right now.',
					request_id: 'r1',
				}),
			})
			.mockResolvedValueOnce(sseResponse(TURN_1_SOURCE, 'First answer [1].')),
	)

	render(<Chat />)

	ask('What is FuelEU?')
	expect(
		await screen.findByText('The model is unavailable right now.'),
	).toBeInTheDocument()

	fireEvent.click(screen.getByRole('button', { name: 'Retry' }))

	expect(await screen.findByText(/First answer/)).toBeInTheDocument()
	expect(
		screen.getByText('The model is unavailable right now.'),
	).toBeInTheDocument()
	expect(screen.getAllByText('What is FuelEU?')).toHaveLength(2)
})
