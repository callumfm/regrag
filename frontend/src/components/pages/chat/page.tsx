import { useRef, useState } from "react"
import {
	MessageScroller,
	MessageScrollerButton,
	MessageScrollerContent,
	MessageScrollerItem,
	MessageScrollerProvider,
	MessageScrollerViewport,
} from "@/components/ui/message-scroller"
import { useChatStream } from "@/hooks/use-chat-stream"
import { numberCitations } from "@/lib/citations"
import { ChatTurn } from "./chat-turn"
import { PromptForm } from "./prompt-form"
import { SourcePanel } from "./source-panel"

type OpenMarker = { turnId: string; marker: number } | null

type TurnHandlers = {
	onOpenMarker: (marker: number) => void
	onRetry: () => void
}

export function ChatPage() {
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
	const openLabel =
		openTurn && openSource
			? (numberCitations(
					openTurn.answer,
					new Set(openTurn.sources.map((source) => source.marker)),
				).get(openSource.marker) ?? null)
			: null

	const isEmpty = turns.length === 0

	return (
		<main className="mx-auto flex h-dvh w-full max-w-3xl flex-col">
			{isEmpty ? (
				<div className="flex flex-1 flex-col justify-end px-6 pb-8">
					<h1 className="text-center font-semibold text-3xl tracking-tight">
						Ask about EU maritime regulation
					</h1>
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
												isBusy={isBusy}
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
			{isEmpty && <div className="flex-1" />}
			<SourcePanel
				source={openSource}
				label={openLabel}
				onClose={() => setOpenMarker(null)}
			/>
		</main>
	)
}
