### 建置階段：使用 Node 建置 Angular 應用程式
FROM node:22-alpine AS build

WORKDIR /usr/src/app

# 先安裝依賴
COPY package*.json ./
RUN npm ci

# 複製專案設定與原始碼
COPY angular.json ./
COPY .postcssrc.json ./
COPY tsconfig*.json ./
COPY src ./src
COPY public ./public

# 預設使用 production 設定進行建置
RUN npm run build


### 執行階段：使用 Nginx 提供靜態檔案，並反向代理 /api 到後端
FROM nginx:1.27-alpine AS runtime

# Angular 20 application builder 預設輸出到 dist/<project-name>/browser
# 注意：要複製「browser 目錄裡的內容」，否則會變成 /usr/share/nginx/html/browser/...
COPY --from=build /usr/src/app/dist/chatbot-ui/browser/ /usr/share/nginx/html/

# 自訂 Nginx 設定（支援 SPA 與 /api 反向代理）
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]


