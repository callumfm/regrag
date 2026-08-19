import { memo } from 'react'
import { Answer } from '@/components/chat/answer'
import { CitedSourcesLine } from '@/components/chat/cited-sources-line'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Bubble, BubbleContent } from '@/components/ui/bubble'
import { Button } from '@/components/ui/button'
import { Message, MessageContent } from '@/components/ui/message'
import type { ChatTurn as Turn } from '@/lib/chat/use-chat-stream'

export const ChatTurn = memo(function ChatTurn({
	turn,
	onOpenMarker,
	onRetry,
}: {
	turn: Turn
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
					{turn.status === 'pending' && (
						<p className="animate-pulse text-muted-foreground text-sm">
							Retrieving…
						</p>
					)}
					{turn.status === 'failed' ? (
						<Alert variant="destructive">
							<AlertTitle>Request failed</AlertTitle>
							<AlertDescription>{turn.error}</AlertDescription>
							<Button
								variant="outline"
								size="sm"
								className="mt-2"
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
							{turn.status === 'settled' && (
								<CitedSourcesLine answer={turn.answer} sources={turn.sources} />
							)}
						</>
					)}
				</MessageContent>
			</Message>
		</div>
	)
})
