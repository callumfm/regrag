import { useRef, useState } from 'react'
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

type TurnHandlers = {
	onOpenMarker: (marker: number) => void
	onRetry: () => void
}

export function Chat() {
	const { turns, ask, stop, isBusy } = useChatStream()
	const [openMarker, setOpenMarker] = useState<OpenMarker>(null)
	const handlersByTurnId = useRef(new Map<string, TurnHandlers>())

	function askQuestion(question: string) {
		setOpenMarker(null)
		ask(question)
	}

	function getTurnHandlers(turnId: string, question: string): TurnHandlers {
		const cached = handlersByTurnId.current.get(turnId)
		if (cached !== undefined) return cached
		const handlers: TurnHandlers = {
			onOpenMarker: (marker) => setOpenMarker({ turnId, marker }),
			onRetry: () => askQuestion(question),
		}
		handlersByTurnId.current.set(turnId, handlers)
		return handlers
	}

	const openTurn = turns.find((turn) => turn.id === openMarker?.turnId)
	const openSource =
		openTurn?.sources.find((source) => source.marker === openMarker?.marker) ??
		null

	return (
		<main className="mx-auto flex h-dvh w-full max-w-3xl flex-col">
			{turns.length === 0 ? (
				<div className="flex flex-1 items-center justify-center p-6">
					<ChatEmpty onSelect={askQuestion} />
				</div>
			) : (
				<MessageScrollerProvider>
					<MessageScroller className="flex-1">
						<MessageScrollerViewport>
							<MessageScrollerContent className="flex flex-col gap-8 px-6 py-6">
								{turns.map((turn) => {
									const handlers = getTurnHandlers(turn.id, turn.question)
									return (
										<MessageScrollerItem
											key={turn.id}
											messageId={turn.id}
											scrollAnchor
										>
											<ChatTurn
												turn={turn}
												onOpenMarker={handlers.onOpenMarker}
												onRetry={handlers.onRetry}
											/>
										</MessageScrollerItem>
									)
								})}
							</MessageScrollerContent>
						</MessageScrollerViewport>
						<MessageScrollerButton />
					</MessageScroller>
				</MessageScrollerProvider>
			)}
			<div className="px-6 pb-6">
				<PromptForm isBusy={isBusy} onSubmit={askQuestion} onStop={stop} />
			</div>
			<SourcePanel source={openSource} onClose={() => setOpenMarker(null)} />
		</main>
	)
}
