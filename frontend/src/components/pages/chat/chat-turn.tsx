import { memo } from "react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Bubble, BubbleContent } from "@/components/ui/bubble"
import { Button } from "@/components/ui/button"
import { Message, MessageContent } from "@/components/ui/message"
import type { ChatTurn as Turn } from "@/hooks/use-chat-stream"
import { Answer } from "./answer"
import { CitedSourcesLine } from "./cited-sources-line"

export const ChatTurn = memo(function ChatTurn({
	turn,
	isBusy,
	onOpenMarker,
	onRetry,
}: {
	turn: Turn
	isBusy: boolean
	onOpenMarker: (marker: number) => void
	onRetry: () => void
}) {
	return (
		<div className="flex flex-col gap-4">
			<Message align="end">
				<MessageContent>
					<Bubble align="end">
						<BubbleContent>{turn.question}</BubbleContent>
					</Bubble>
				</MessageContent>
			</Message>
			<Message align="start">
				<MessageContent>
					{turn.status === "pending" && (
						<p className="animate-pulse text-muted-foreground text-sm">
							Retrieving…
						</p>
					)}
					{turn.status === "failed" ? (
						<Alert variant="destructive">
							<AlertTitle>Request failed</AlertTitle>
							<AlertDescription>{turn.error}</AlertDescription>
							<Button
								variant="outline"
								size="sm"
								className="mt-2"
								disabled={isBusy}
								onClick={onRetry}
							>
								Retry
							</Button>
						</Alert>
					) : (
						<>
							<Answer
								answer={turn.answer}
								sources={turn.sources}
								onOpenMarker={onOpenMarker}
							/>
							{turn.status === "settled" && (
								<CitedSourcesLine answer={turn.answer} sources={turn.sources} />
							)}
						</>
					)}
				</MessageContent>
			</Message>
		</div>
	)
})
