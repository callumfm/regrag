import { Button } from '@/components/ui/button'
import {
	Empty,
	EmptyContent,
	EmptyDescription,
	EmptyHeader,
	EmptyTitle,
} from '@/components/ui/empty'

const SUGGESTIONS = [
	'What is the greenhouse gas intensity limit under FuelEU Maritime?',
	'When must a company monitor energy used on board?',
	'Which ships fall outside the scope of the regulation?',
]

export function ChatEmpty({
	onSelect,
}: {
	onSelect: (question: string) => void
}) {
	return (
		<Empty>
			<EmptyHeader>
				<EmptyTitle>Ask about EU regulation</EmptyTitle>
				<EmptyDescription>
					Answers are drawn from the indexed corpus and cite the paragraphs they
					rest on.
				</EmptyDescription>
			</EmptyHeader>
			<EmptyContent>
				<div className="flex flex-col gap-2">
					{SUGGESTIONS.map((suggestion) => (
						<Button
							key={suggestion}
							variant="outline"
							size="sm"
							onClick={() => onSelect(suggestion)}
						>
							{suggestion}
						</Button>
					))}
				</div>
			</EmptyContent>
		</Empty>
	)
}
