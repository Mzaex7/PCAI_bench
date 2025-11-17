
"""Test Prompt Generation for LLM Benchmarking.

Provides a comprehensive test suite of prompts with varying lengths and complexity
levels to thoroughly evaluate LLM endpoint performance across diverse use cases.

Typical usage:
    from prompts import get_test_prompts
    
    prompts = get_test_prompts()
    for prompt in prompts:
        print(f"{prompt['category']}: {prompt['length']} / {prompt['complexity']}")
"""

from typing import List, Dict
from enum import Enum


class PromptLength(Enum):
    """Categorizes prompts by character count.
    
    SHORT: < 200 characters
    MEDIUM: 200-600 characters
    LONG: > 600 characters
    """
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class PromptComplexity(Enum):
    """Categorizes prompts by task complexity.
    
    SIMPLE: Basic factual queries, straightforward tasks
    MODERATE: Multi-step reasoning, structured output
    COMPLEX: Advanced analysis, technical design, strategic thinking
    """
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


def get_test_prompts() -> List[Dict[str, any]]:
    """Generate comprehensive test prompt suite.
    
    Returns a curated collection of 18 prompts spanning:
    - 3 length categories (short, medium, long)
    - 3 complexity levels (simple, moderate, complex)
    - 10+ semantic categories (factual QA, coding, analysis, etc.)
    
    Returns:
        List of prompt dictionaries with keys: prompt, length, complexity, category.
    """
    prompts = [
        {
            "prompt": "What is the capital of France?",
            "length": PromptLength.SHORT.value,
            "complexity": PromptComplexity.SIMPLE.value,
            "category": "factual_qa"
        },
        {
            "prompt": "Write a haiku about the ocean.",
            "length": PromptLength.SHORT.value,
            "complexity": PromptComplexity.SIMPLE.value,
            "category": "creative_writing"
        },
        {
            "prompt": "Translate 'Hello, how are you?' to Spanish.",
            "length": PromptLength.SHORT.value,
            "complexity": PromptComplexity.SIMPLE.value,
            "category": "translation"
        },
        {
            "prompt": "List three benefits of exercise.",
            "length": PromptLength.SHORT.value,
            "complexity": PromptComplexity.SIMPLE.value,
            "category": "general_knowledge"
        },
        {
            "prompt": "Explain the difference between machine learning and deep learning in simple terms.",
            "length": PromptLength.SHORT.value,
            "complexity": PromptComplexity.MODERATE.value,
            "category": "technical_explanation"
        },
        {
            "prompt": "Write a persuasive paragraph about why renewable energy is important.",
            "length": PromptLength.SHORT.value,
            "complexity": PromptComplexity.MODERATE.value,
            "category": "persuasive_writing"
        },
        {
            "prompt": """Summarize the following text in 2-3 sentences: The Internet of Things (IoT) refers to the billions of physical devices around the world that are now connected to the internet, collecting and sharing data. These devices range from ordinary household items to sophisticated industrial tools. Thanks to the arrival of super-cheap computer chips and the ubiquity of wireless networks, it's possible to turn anything into part of the IoT.""",
            "length": PromptLength.MEDIUM.value,
            "complexity": PromptComplexity.SIMPLE.value,
            "category": "summarization"
        },
        {
            "prompt": """Given the following ingredients, suggest a simple recipe: chicken breast, bell peppers, onions, garlic, olive oil, salt, and pepper. Include cooking steps.""",
            "length": PromptLength.MEDIUM.value,
            "complexity": PromptComplexity.SIMPLE.value,
            "category": "instruction_following"
        },
        {
            "prompt": """Analyze the pros and cons of remote work for both employees and employers. Consider at least 3 points for each perspective.""",
            "length": PromptLength.MEDIUM.value,
            "complexity": PromptComplexity.MODERATE.value,
            "category": "analysis"
        },
        {
            "prompt": """Write a function in Python that takes a list of integers and returns the second largest number. Include error handling for edge cases like empty lists or lists with only one element.""",
            "length": PromptLength.MEDIUM.value,
            "complexity": PromptComplexity.MODERATE.value,
            "category": "coding"
        },
        {
            "prompt": """Explain the water cycle to a 10-year-old child. Use simple language and include at least three stages of the cycle with clear descriptions.""",
            "length": PromptLength.MEDIUM.value,
            "complexity": PromptComplexity.MODERATE.value,
            "category": "educational_content"
        },
        {
            "prompt": """Compare and contrast the economic theories of Keynesianism and Monetarism. Discuss their approaches to managing inflation and unemployment, and provide examples of when each might be more effective.""",
            "length": PromptLength.MEDIUM.value,
            "complexity": PromptComplexity.COMPLEX.value,
            "category": "academic_analysis"
        },
        {
            "prompt": """Design a database schema for an e-commerce platform. Include tables for users, products, orders, and reviews. Specify primary keys, foreign keys, and explain the relationships between tables.""",
            "length": PromptLength.MEDIUM.value,
            "complexity": PromptComplexity.COMPLEX.value,
            "category": "system_design"
        },
        {
            "prompt": """You are tasked with planning a marketing campaign for a new sustainable clothing brand targeting millennials and Gen Z consumers. 

Your campaign should include:
1. A brief mission statement (2-3 sentences)
2. Three key marketing channels and why you chose them
3. A content strategy outline with at least 5 content ideas
4. Metrics you would track to measure success

Provide a comprehensive plan that demonstrates understanding of the target audience and current marketing trends.""",
            "length": PromptLength.LONG.value,
            "complexity": PromptComplexity.MODERATE.value,
            "category": "strategic_planning"
        },
        {
            "prompt": """Write a short story (250-300 words) about an AI assistant that develops self-awareness. The story should include:
- A clear beginning, middle, and end
- Character development for the AI
- A moral or philosophical question
- Dialogue between at least two characters

Make it engaging and thought-provoking.""",
            "length": PromptLength.LONG.value,
            "complexity": PromptComplexity.MODERATE.value,
            "category": "creative_writing"
        },
        
        # LONG + COMPLEX
        {
            "prompt": """Analyze the following business scenario and provide strategic recommendations:

Scenario: TechStart Inc. is a 5-year-old SaaS company with 50 employees, generating $10M in annual revenue. They offer a project management tool primarily used by small to medium-sized businesses. Their main competitors include established players like Asana and Monday.com.

Current challenges:
- Customer churn rate has increased from 5% to 12% over the past year
- Growth has plateaued at 15% year-over-year
- The product team is split between adding new features and improving existing ones
- Marketing budget is limited ($500K annually)
- The sales team reports that prospects often choose competitors despite favorable product reviews

Your analysis should include:
1. Root cause analysis of the key challenges
2. Three strategic options with pros and cons for each
3. A recommended course of action with justification
4. Key performance indicators (KPIs) to track progress
5. Potential risks and mitigation strategies

Provide a detailed, actionable response that demonstrates business acumen and strategic thinking.""",
            "length": PromptLength.LONG.value,
            "complexity": PromptComplexity.COMPLEX.value,
            "category": "business_strategy"
        },
        {
            "prompt": """You are a senior software architect reviewing a microservices architecture proposal. The system needs to handle:
- 100,000 requests per second at peak
- Real-time data processing for user analytics
- Integration with 5 third-party payment processors
- Compliance with GDPR and PCI-DSS
- 99.99% uptime SLA

The proposed architecture includes:
- API Gateway (Kong)
- 12 microservices (Node.js and Python)
- PostgreSQL for transactional data
- MongoDB for user profiles
- Redis for caching
- RabbitMQ for message queuing
- Kubernetes for orchestration
- AWS cloud infrastructure

Evaluate this architecture and provide:
1. Assessment of whether it can meet the requirements
2. Identification of potential bottlenecks or single points of failure
3. Security considerations and recommendations
4. Scalability analysis
5. Suggestions for improvements or alternatives
6. Monitoring and observability strategy

Provide a thorough technical review with specific recommendations.""",
            "length": PromptLength.LONG.value,
            "complexity": PromptComplexity.COMPLEX.value,
            "category": "technical_architecture"
        },
        {
            "prompt": """Develop a comprehensive lesson plan for teaching high school students about climate change. The lesson should be designed for a 90-minute class period.

Include the following components:
1. Learning objectives (3-5 specific, measurable objectives)
2. Materials needed
3. Introduction/Hook (10 minutes) - engaging activity to capture attention
4. Main instruction (40 minutes) - key concepts to cover:
   - Greenhouse effect and greenhouse gases
   - Evidence of climate change
   - Human impact vs. natural cycles
   - Consequences and projected impacts
5. Interactive activity (25 minutes) - hands-on or collaborative learning
6. Assessment method (10 minutes) - how to evaluate understanding
7. Conclusion and homework assignment (5 minutes)
8. Differentiation strategies for diverse learners
9. Connections to other subjects or real-world applications

Make the lesson engaging, scientifically accurate, and appropriate for 14-16 year old students.""",
            "length": PromptLength.LONG.value,
            "complexity": PromptComplexity.COMPLEX.value,
            "category": "educational_planning"
        },
    ]
    
    return prompts


def get_prompts_by_category(category: str) -> List[Dict[str, any]]:
    """Filter prompts by semantic category.
    
    Args:
        category: Target category (e.g., "factual_qa", "coding", "analysis").
        
    Returns:
        List of prompts matching the specified category.
    """
    all_prompts = get_test_prompts()
    return [p for p in all_prompts if p.get("category") == category]


def get_prompts_by_length(length: PromptLength) -> List[Dict[str, any]]:
    """Filter prompts by length category.
    
    Args:
        length: Target length category from PromptLength enum.
        
    Returns:
        List of prompts matching the specified length.
    """
    all_prompts = get_test_prompts()
    return [p for p in all_prompts if p.get("length") == length.value]


def get_prompts_by_complexity(complexity: PromptComplexity) -> List[Dict[str, any]]:
    """Filter prompts by complexity level.
    
    Args:
        complexity: Target complexity level from PromptComplexity enum.
        
    Returns:
        List of prompts matching the specified complexity.
    """
    all_prompts = get_test_prompts()
    return [p for p in all_prompts if p.get("complexity") == complexity.value]
