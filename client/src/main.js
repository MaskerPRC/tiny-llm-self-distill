import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'

import Dashboard from './views/Dashboard.vue'
import Chat from './views/Chat.vue'
import Tools from './views/Tools.vue'
import Evolution from './views/Evolution.vue'
import Loop from './views/Loop.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: Dashboard },
    { path: '/chat', component: Chat },
    { path: '/tools', component: Tools },
    { path: '/evolution', component: Evolution },
    { path: '/loop', component: Loop },
  ],
})

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')
