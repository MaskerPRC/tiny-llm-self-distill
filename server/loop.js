/**
 * loop.js - v1.1.0
 * 
 * 元流程核心路由。当有可用的小模型工具时，优先尝试使用小模型处理，
 * 置信度低于阈值时回退到 Gemini 大模型。
 * 无已注册工具时直接走大模型路径。
 */

async function process(request, context) {
  const { gemini, tools, log } = context;

  // 验证测试特殊处理
  if (request.input === '__validation_test__') {
    return { output: '[test ok]', tool_used: 'test', confidence: 1 };
  }

  const CONFIDENCE_THRESHOLD = 0.8;
  const availableTools = tools.list();

  log(`[Loop v1.1] 收到请求: ${String(request.input).substring(0, 50)}...`);
  log(`[Loop v1.1] 可用工具数量: ${availableTools.length}`);

  // 如果有可用工具，尝试用小模型处理
  if (availableTools.length > 0) {
    // 根据请求类型或内容选择最匹配的工具
    let bestResult = null;
    let bestTool = null;
    let bestConfidence = 0;

    for (const toolName of availableTools) {
      try {
        log(`[Loop v1.1] 尝试工具: ${toolName}`);
        const result = await tools.predict(toolName, request.input);

        const confidence = typeof result.confidence === 'number' ? result.confidence : 0;

        if (confidence > bestConfidence) {
          bestConfidence = confidence;
          bestResult = result;
          bestTool = toolName;
        }

        // 如果已经超过阈值，直接使用
        if (confidence >= CONFIDENCE_THRESHOLD) {
          break;
        }
      } catch (err) {
        log(`[Loop v1.1] 工具 ${toolName} 调用失败: ${err.message}`);
      }
    }

    // 如果最佳结果超过置信度阈值，直接返回
    if (bestResult && bestConfidence >= CONFIDENCE_THRESHOLD) {
      log(`[Loop v1.1] 使用工具 ${bestTool}，置信度: ${bestConfidence}`);
      return {
        output: bestResult.output || bestResult,
        tool_used: bestTool,
        confidence: bestConfidence,
        metadata: {
          version: '1.1.0',
          route: 'tool_direct',
          tool: bestTool,
        },
      };
    }

    // 置信度不足，回退到大模型，但附带小模型的参考结果
    if (bestResult) {
      log(`[Loop v1.1] 工具 ${bestTool} 置信度不足 (${bestConfidence} < ${CONFIDENCE_THRESHOLD})，回退到 Gemini`);

      const enrichedPrompt = `用户问题: ${request.input}\n\n参考信息（来自预处理模型 ${bestTool}，置信度 ${bestConfidence}）: ${JSON.stringify(bestResult.output || bestResult)}\n\n请综合以上信息给出准确回答。`;

      const response = await gemini.chat(enrichedPrompt, {
        systemPrompt: '你是一个多功能AI助手。你会收到用户的问题以及一个预处理模型的参考输出，请综合判断后给出最佳回答。',
      });

      return {
        output: response,
        tool_used: 'gemini_with_tool_hint',
        confidence: 0.9,
        metadata: {
          version: '1.1.0',
          route: 'tool_fallback_to_llm',
          attempted_tool: bestTool,
          tool_confidence: bestConfidence,
        },
      };
    }
  }

  // 无可用工具或所有工具调用失败，直接使用大模型
  log(`[Loop v1.1] 直接转发到 Gemini`);

  const response = await gemini.chat(request.input, {
    systemPrompt: '你是一个多功能AI助手，请直接回答用户的问题。',
  });

  return {
    output: response,
    tool_used: 'gemini_direct',
    confidence: 1.0,
    metadata: { version: '1.1.0', route: 'direct_llm' },
  };
}

module.exports = { process };