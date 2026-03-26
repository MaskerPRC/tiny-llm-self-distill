/**
 * loop.js - 初始版本 (v1.0.0)
 * 
 * 这是元流程的起始状态：所有请求直接转发给 Gemini 3.1 Pro 处理。
 * 随着系统运行，evolver 会分析请求模式，训练小模型，
 * 并使用 Claude Opus 4.6 生成新版本的 loop.js，插入前置分流逻辑。
 * 
 * 导出函数签名: async process(request, context) => result
 * - request: { input, type?, metadata? }
 * - context: { gemini, tools, log }
 * - result:  { output, tool_used, confidence, metadata? }
 */

async function process(request, context) {
  const { gemini, log } = context;

  log(`[Loop v1] 直接转发到 Gemini: ${request.input.substring(0, 50)}...`);

  const response = await gemini.chat(request.input, {
    systemPrompt: '你是一个多功能AI助手，请直接回答用户的问题。',
  });

  return {
    output: response,
    tool_used: 'gemini_direct',
    confidence: 1.0,
    metadata: { version: '1.0.0', route: 'direct_llm' },
  };
}

module.exports = { process };
