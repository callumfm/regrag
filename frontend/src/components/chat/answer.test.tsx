import { fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { expect, test } from 'vitest'
import type { ChatSource } from '@/lib/chat/use-chat-stream'
import { Answer } from './answer'
import { SourcePanel } from './source-panel'

const SOURCES: ChatSource[] = [
	{
		marker: 1,
		chunk_id: 1,
		celex: '32023R1805',
		citation: 'Article 4(1)',
		title: 'Greenhouse gas intensity limit',
		text: 'The greenhouse gas intensity of the energy used on board.',
	},
	{
		marker: 2,
		chunk_id: 2,
		celex: '32023R1805',
		citation: 'Article 5(1)',
		title: null,
		text: 'Companies shall monitor the energy used on board.',
	},
]

function AnswerWithPanel({ answer }: { answer: string }) {
	const [open, setOpen] = useState<number | null>(null)
	return (
		<>
			<Answer answer={answer} sources={SOURCES} onOpenMarker={setOpen} />
			<SourcePanel
				source={SOURCES.find((source) => source.marker === open) ?? null}
				onClose={() => setOpen(null)}
			/>
		</>
	)
}

test('opens the paragraph behind the citation that was clicked', async () => {
	render(<AnswerWithPanel answer="Ships must comply [2]." />)
	fireEvent.click(screen.getByRole('button', { name: 'Open source 2' }))
	expect(
		await screen.findByText(
			'Companies shall monitor the energy used on board.',
		),
	).toBeInTheDocument()
})

test('renders a marker with no matching source as plain text', () => {
	render(<AnswerWithPanel answer="Invented [7]." />)
	expect(screen.queryByRole('button', { name: /Open source/ })).toBeNull()
	expect(screen.getByText(/Invented \[7\]\./)).toBeInTheDocument()
})

test('splits two distinct citation markers in a single answer, each opening its own source', async () => {
	render(<AnswerWithPanel answer="Ships must comply [1] and monitor [2]." />)

	fireEvent.click(screen.getByRole('button', { name: 'Open source 1' }))
	expect(
		await screen.findByText(
			'The greenhouse gas intensity of the energy used on board.',
		),
	).toBeInTheDocument()

	fireEvent.click(screen.getByRole('button', { name: 'Open source 2' }))
	expect(
		await screen.findByText(
			'Companies shall monitor the energy used on board.',
		),
	).toBeInTheDocument()
})
