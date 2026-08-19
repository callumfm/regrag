import {
	Drawer,
	DrawerContent,
	DrawerDescription,
	DrawerHeader,
	DrawerTitle,
} from '@/components/ui/drawer'
import type { ChatSource } from '@/lib/chat/use-chat-stream'

export function SourcePanel({
	source,
	onClose,
}: {
	source: ChatSource | null
	onClose: () => void
}) {
	return (
		<Drawer
			open={source !== null}
			onOpenChange={(open: boolean) => {
				if (!open) onClose()
			}}
			swipeDirection="right"
		>
			<DrawerContent>
				<DrawerHeader>
					<DrawerTitle>{source?.citation}</DrawerTitle>
					<DrawerDescription>
						{source?.celex}
						{source?.title ? ` · ${source.title}` : ''}
					</DrawerDescription>
				</DrawerHeader>
				<div className="flex-1 overflow-y-auto p-4 text-sm leading-relaxed whitespace-pre-wrap">
					{source?.text}
				</div>
			</DrawerContent>
		</Drawer>
	)
}
