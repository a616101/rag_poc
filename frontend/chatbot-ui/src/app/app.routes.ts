import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    redirectTo: 'ask-stream'
  },
  {
    path: 'ask-stream',
    loadComponent: () =>
      import('./pages/chat-page/chat-page.component').then(
        (m) => m.ChatPageComponent
      ),
    data: {
      title: 'Unified Agent 串流（/ask/stream）',
      apiPath: '/api/v1/rag/ask/stream'
    }
  },
  {
    path: 'ask-stream-chat',
    loadComponent: () =>
      import('./pages/chat-page/chat-page.component').then(
        (m) => m.ChatPageComponent
      ),
    data: {
      title: 'Unified Agent 串流（/ask/stream_chat）',
      apiPath: '/api/v1/rag/ask/stream_chat'
    }
  },
  {
    path: '**',
    redirectTo: 'ask-stream'
  }
];
