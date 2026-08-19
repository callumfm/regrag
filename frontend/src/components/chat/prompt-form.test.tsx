import { fireEvent, render, screen } from '@testing-library/react'
import { expect, test, vi } from 'vitest'
import { PromptForm } from './prompt-form'

function noop() {}

test('submits the trimmed question and clears the box', () => {
	const onSubmit = vi.fn()
	render(<PromptForm isBusy={false} onSubmit={onSubmit} onStop={noop} />)
	const box = screen.getByRole('textbox')
	fireEvent.change(box, { target: { value: '  What is FuelEU?  ' } })
	fireEvent.click(screen.getByRole('button', { name: 'Send' }))
	expect(onSubmit).toHaveBeenCalledWith('What is FuelEU?')
	expect(box).toHaveValue('')
})

test('does not submit an empty question', () => {
	const onSubmit = vi.fn()
	render(<PromptForm isBusy={false} onSubmit={onSubmit} onStop={noop} />)
	fireEvent.click(screen.getByRole('button', { name: 'Send' }))
	expect(onSubmit).not.toHaveBeenCalled()
})

test('offers stop instead of send while streaming', () => {
	const onStop = vi.fn()
	render(<PromptForm isBusy={true} onSubmit={noop} onStop={onStop} />)
	fireEvent.click(screen.getByRole('button', { name: 'Stop' }))
	expect(onStop).toHaveBeenCalled()
	expect(screen.queryByRole('button', { name: 'Send' })).toBeNull()
})
