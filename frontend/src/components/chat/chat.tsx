import { useState } from 'react'
import { ChatEmpty } from '@/components/chat/chat-empty'
import { ChatTurn } from '@/components/chat/chat-turn'
import { PromptForm } from '@/components/chat/prompt-form'
import { SourcePanel } from '@/components/chat/source-panel'
import {
	MessageScroller,
	MessageScrollerButton,
	MessageScrollerContent,
	MessageScrollerItem,
	MessageScrollerProvider,
	MessageScrollerViewport,
} from '@/components/ui/message-scroller'
import { useChatStream } from '@/lib/chat/use-chat-stream'

type OpenMarker = { turnId: string; marker: number } | null

export function Chat() {
	const { turns, ask, stop, isBusy } = useChatStream()
	const [openMarker, setOpenMarker] = useState<OpenMarker>(null)

	const openTurn = turns.find((turn) => turn.id === openMarker?.turnId)
	const openSource =
		openTurn?.sources.find((source) => source.marker === openMarker?.marker) ??
		null

	return (
		<main className="mx-auto flex h-screen w-full max-w-3xl flex-col">
			{turns.length === 0 ? (
				<div className="flex flex-1 items-center justify-center p-6">
					<ChatEmpty onSelect={ask} />
				</div>
			) : (
				<MessageScrollerProvider>
					<MessageScroller className="flex-1">
						<MessageScrollerViewport>
							<MessageScrollerContent className="flex flex-col gap-8 px-6 py-6">
								{turns.map((turn) => (
									<MessageScrollerItem
										key={turn.id}
										messageId={turn.id}
										scrollAnchor
									>
										<ChatTurn
											turn={turn}
											onOpenMarker={(marker) =>
												setOpenMarker({ turnId: turn.id, marker })
											}
										/>
									</MessageScrollerItem>
								))}
							</MessageScrollerContent>
						</MessageScrollerViewport>
						<MessageScrollerButton />
					</MessageScroller>
				</MessageScrollerProvider>
			)}
			<div className="px-6 pb-6">
				<PromptForm isBusy={isBusy} onSubmit={ask} onStop={stop} />
			</div>
			<SourcePanel source={openSource} onClose={() => setOpenMarker(null)} />
		</main>
	)
}
