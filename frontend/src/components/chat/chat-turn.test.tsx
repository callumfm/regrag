import { fireEvent, render, screen } from '@testing-library/react'
import { expect, test, vi } from 'vitest'
import type { ChatSource, ChatTurn as Turn } from '@/lib/chat/use-chat-stream'
import { ChatTurn } from './chat-turn'

const SOURCE: ChatSource = {
	marker: 1,
	chunk_id: 1,
	celex: '32023R1805',
	citation: 'Article 4(1)',
	title: 'Greenhouse gas intensity limit',
	text: 'The greenhouse gas intensity of the energy used on board.',
}

function turn(overrides: Partial<Turn> = {}): Turn {
	return {
		id: 'turn-1',
		question: 'What is FuelEU?',
		answer: '',
		sources: [],
		status: 'pending',
		error: null,
		...overrides,
	}
}

function renderTurn(value: Turn, onRetry: () => void = () => {}) {
	render(<ChatTurn turn={value} onOpenMarker={() => {}} onRetry={onRetry} />)
}

test('shows retrieval progress before any source arrives', () => {
	renderTurn(turn())
	expect(screen.getByText('Retrieving…')).toBeInTheDocument()
})

test('counts the cited sources against everything retrieved', () => {
	renderTurn(
		turn({
			answer: 'Ships must comply [1].',
			sources: [SOURCE, { ...SOURCE, marker: 2, chunk_id: 2 }],
			status: 'settled',
		}),
	)
	expect(
		screen.getByText('Cited 1 of 2 retrieved passages'),
	).toBeInTheDocument()
})

test('a refusal shows no sources line', () => {
	renderTurn(turn({ answer: 'I cannot answer that.', status: 'settled' }))
	expect(screen.queryByText(/retrieved passages/)).toBeNull()
	expect(screen.getByText('I cannot answer that.')).toBeInTheDocument()
})

test('a failed turn shows the error instead of an answer', () => {
	renderTurn(turn({ status: 'failed', error: 'chat call failed' }))
	expect(screen.getByText('chat call failed')).toBeInTheDocument()
})

test('a failed turn offers a retry that re-asks the question', () => {
	const onRetry = vi.fn()
	renderTurn(turn({ status: 'failed', error: 'chat call failed' }), onRetry)
	fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
	expect(onRetry).toHaveBeenCalled()
})
