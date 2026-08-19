import { ArrowUpIcon, SquareIcon } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'

export function PromptForm({
	isBusy,
	onSubmit,
	onStop,
}: {
	isBusy: boolean
	onSubmit: (question: string) => void
	onStop: () => void
}) {
	const [question, setQuestion] = useState('')

	function submit(event: { preventDefault: () => void }) {
		event.preventDefault()
		const trimmed = question.trim()
		if (trimmed === '') return
		setQuestion('')
		onSubmit(trimmed)
	}

	return (
		<form
			onSubmit={submit}
			className="flex items-end gap-2 rounded-2xl border p-2"
		>
			<Textarea
				value={question}
				onChange={(event) => setQuestion(event.target.value)}
				onKeyDown={(event) => {
					if (event.key === 'Enter' && !event.shiftKey) submit(event)
				}}
				placeholder="Ask about EU regulation…"
				rows={1}
				className="min-h-9 flex-1 resize-none border-0 bg-transparent focus-visible:ring-0"
			/>
			{isBusy ? (
				<Button type="button" size="icon" aria-label="Stop" onClick={onStop}>
					<SquareIcon />
				</Button>
			) : (
				<Button type="submit" size="icon" aria-label="Send">
					<ArrowUpIcon />
				</Button>
			)}
		</form>
	)
}
