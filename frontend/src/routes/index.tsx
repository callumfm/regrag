import { createFileRoute } from "@tanstack/react-router"
import { ChatPage } from "@/components/pages/chat/page"

export const Route = createFileRoute("/")({ component: ChatPage })
