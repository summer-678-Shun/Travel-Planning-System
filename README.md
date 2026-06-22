## 启动后端
#### 1. 进入后端目录
cd helloagents-trip-planner/backend

#### 2. 安装依赖
pip install -r requirements.txt

#### 3. 配置环境变量
cp .env.example .env
#### 编辑.env文件，填入你的API密钥

#### 4. 启动后端服务
uvicorn app.api.main:app --reload
#### 或者
python run.py

## 启动前端
#### 1. 进入前端目录
cd helloagents-trip-planner/frontend

#### 2. 安装依赖
npm install

#### 3. 启动前端服务
npm run dev
