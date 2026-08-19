import { useMemo } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { CitationChip } from '@/components/chat/citation-chip'
import { rehypeCitationMarkers } from '@/lib/chat/markers'
import type { ChatSource } from '@/lib/chat/use-chat-stream'

const PROSE =
	'text-sm leading-relaxed [&_p]:mb-3 [&_ul]:mb-3 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:mb-3 [&_ol]:list-decimal [&_ol]:pl-5 [&_strong]:font-semibold [&_code]:rounded [&_code]:bg-muted [&_code]:px-1'

export function Answer({
	answer,
	sources,
	onOpenMarker,
}: {
	answer: string
	sources: ChatSource[]
	onOpenMarker: (marker: number) => void
}) {
	const known = useMemo(
		() => new Set(sources.map((source) => source.marker)),
		[sources],
	)
	const components = useMemo(
		() =>
			({
				'cite-marker': ({ marker }: { marker?: string }) => (
					<CitationChip marker={Number(marker)} onOpen={onOpenMarker} />
				),
			}) as Components,
		[onOpenMarker],
	)

	return (
		<div className={PROSE}>
			<ReactMarkdown
				remarkPlugins={[remarkGfm]}
				rehypePlugins={[rehypeCitationMarkers(known)]}
				components={components}
			>
				{answer}
			</ReactMarkdown>
		</div>
	)
}
