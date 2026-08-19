import { extractCitedMarkers } from '@/lib/chat/markers'
import type { ChatSource } from '@/lib/chat/use-chat-stream'

export function CitedSourcesLine({
	answer,
	sources,
}: {
	answer: string
	sources: ChatSource[]
}) {
	const known = new Set(sources.map((source) => source.marker))
	const cited = extractCitedMarkers(answer, known)
	if (cited.length === 0) return null
	return (
		<p className="text-muted-foreground text-xs">
			Cited {cited.length} of {sources.length} retrieved passages
		</p>
	)
}
