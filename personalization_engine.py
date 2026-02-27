"""
Personalization Recommendation Engine - 个性化推荐引擎
基于用户理解系统的推荐算法实现
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
import math
from collections import defaultdict
import json


@dataclass
class ContentItem:
    """内容项"""
    id: str
    title: str
    content_type: str  # article/code/tool/concept
    topics: List[str]
    difficulty: str  # beginner/intermediate/advanced/expert
    estimated_time: int  # 分钟
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "content_type": self.content_type,
            "topics": self.topics,
            "difficulty": self.difficulty,
            "estimated_time": self.estimated_time,
            "metadata": self.metadata
        }


@dataclass
class Recommendation:
    """推荐结果"""
    item: ContentItem
    score: float
    reason: str
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            "item": self.item.to_dict(),
            "score": round(self.score, 3),
            "reason": self.reason,
            "confidence": round(self.confidence, 2)
        }


class UserInterestModel:
    """
    用户兴趣模型
    基于显式和隐式反馈构建动态兴趣图谱
    """
    
    def __init__(self):
        # 主题兴趣分数 (-1 到 1, 0为中性)
        self.topic_scores: Dict[str, float] = defaultdict(float)
        
        # 兴趣演化历史
        self.topic_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # 内容类型偏好
        self.content_type_scores: Dict[str, float] = defaultdict(lambda: 0.5)
        
        # 难度偏好
        self.difficulty_preference: str = "adaptive"
        
        # 探索-利用平衡
        self.exploration_rate: float = 0.2
        
    def update_from_interaction(self, content: ContentItem, 
                                interaction_type: str,
                                duration: Optional[int] = None):
        """
        从交互更新兴趣模型
        
        Args:
            content: 内容项
            interaction_type: view/click/save/share/complete/ignore
            duration: 交互时长(秒)
        """
        # 计算反馈权重
        weight = self._get_interaction_weight(interaction_type, duration)
        
        # 更新主题分数
        for topic in content.topics:
            old_score = self.topic_scores[topic]
            
            # 应用时间衰减的学习率
            learning_rate = 0.1
            new_score = old_score + learning_rate * weight * (1 - abs(old_score))
            
            # 限制范围
            self.topic_scores[topic] = max(-1, min(1, new_score))
            
            # 记录历史
            self.topic_history[topic].append((datetime.now(), self.topic_scores[topic]))
        
        # 更新内容类型偏好
        type_weight = 0.05
        old_type_score = self.content_type_scores[content.content_type]
        self.content_type_scores[content.content_type] = old_type_score + type_weight * weight
    
    def _get_interaction_weight(self, interaction_type: str, 
                                 duration: Optional[int]) -> float:
        """获取交互类型权重"""
        weights = {
            "ignore": -0.5,
            "view": 0.1,
            "click": 0.3,
            "save": 0.6,
            "share": 0.8,
            "complete": 1.0,
            "dismiss": -0.3
        }
        
        base_weight = weights.get(interaction_type, 0)
        
        # 根据时长调整
        if duration and interaction_type in ["view", "complete"]:
            if duration > 300:  # 超过5分钟
                base_weight *= 1.2
            elif duration < 10:  # 少于10秒
                base_weight *= 0.3
        
        return base_weight
    
    def get_top_interests(self, n: int = 10) -> List[Tuple[str, float]]:
        """获取前N个兴趣主题"""
        sorted_topics = sorted(
            self.topic_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_topics[:n]
    
    def get_interest_trend(self, topic: str, days: int = 30) -> str:
        """获取兴趣趋势"""
        history = self.topic_history.get(topic, [])
        if len(history) < 2:
            return "stable"
        
        # 获取最近的数据
        cutoff = datetime.now() - timedelta(days=days)
        recent = [h for h in history if h[0] > cutoff]
        
        if len(recent) < 2:
            return "stable"
        
        # 计算趋势
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        first_avg = sum(h[1] for h in first_half) / len(first_half)
        second_avg = sum(h[1] for h in second_half) / len(second_half)
        
        diff = second_avg - first_avg
        if diff > 0.1:
            return "rising"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"
    
    def calculate_novelty_score(self, content: ContentItem) -> float:
        """
        计算内容的新颖性分数
        基于与用户已知兴趣的差异
        """
        # 计算与现有兴趣的相似度
        similarity = 0
        for topic in content.topics:
            similarity += abs(self.topic_scores[topic])
        
        similarity /= max(len(content.topics), 1)
        
        # 新颖性 = 1 - 相似度
        novelty = 1 - similarity
        
        return novelty


class CollaborativeFilter:
    """
    协同过滤模块
    基于相似用户的偏好进行推荐
    """
    
    def __init__(self):
        # 用户-内容交互矩阵
        self.user_item_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # 用户相似度缓存
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
    def add_interaction(self, user_id: str, item_id: str, rating: float):
        """添加用户-内容交互"""
        self.user_item_matrix[user_id][item_id] = rating
        # 清除相关缓存
        self._clear_similarity_cache(user_id)
    
    def _clear_similarity_cache(self, user_id: str):
        """清除用户相似度缓存"""
        keys_to_remove = [k for k in self.similarity_cache.keys() if user_id in k]
        for key in keys_to_remove:
            del self.similarity_cache[key]
    
    def calculate_user_similarity(self, user1: str, user2: str) -> float:
        """计算两个用户的相似度 (余弦相似度)"""
        cache_key = tuple(sorted([user1, user2]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        items1 = self.user_item_matrix[user1]
        items2 = self.user_item_matrix[user2]
        
        # 找到共同评价的内容
        common_items = set(items1.keys()) & set(items2.keys())
        
        if not common_items:
            return 0.0
        
        # 计算余弦相似度
        dot_product = sum(items1[item] * items2[item] for item in common_items)
        
        norm1 = math.sqrt(sum(r ** 2 for r in items1.values()))
        norm2 = math.sqrt(sum(r ** 2 for r in items2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        self.similarity_cache[cache_key] = similarity
        
        return similarity
    
    def get_similar_users(self, user_id: str, n: int = 5) -> List[Tuple[str, float]]:
        """获取最相似的N个用户"""
        similarities = []
        
        for other_user in self.user_item_matrix.keys():
            if other_user != user_id:
                sim = self.calculate_user_similarity(user_id, other_user)
                if sim > 0:
                    similarities.append((other_user, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """预测用户对内容的评分"""
        if item_id in self.user_item_matrix[user_id]:
            return self.user_item_matrix[user_id][item_id]
        
        # 基于相似用户的评分预测
        similar_users = self.get_similar_users(user_id, n=10)
        
        weighted_sum = 0
        similarity_sum = 0
        
        for other_user, similarity in similar_users:
            if item_id in self.user_item_matrix[other_user]:
                rating = self.user_item_matrix[other_user][item_id]
                weighted_sum += similarity * rating
                similarity_sum += similarity
        
        if similarity_sum == 0:
            return 0.5  # 默认中性评分
        
        return weighted_sum / similarity_sum


class ContextualBandit:
    """
    上下文多臂老虎机
    用于探索-利用平衡和实时推荐优化
    """
    
    def __init__(self, n_arms: int = 10):
        self.n_arms = n_arms
        
        # 每个臂的统计
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        
        # 上下文特征权重
        self.context_weights: Dict[str, List[float]] = defaultdict(
            lambda: [0.0] * n_arms
        )
        
    def select_arm(self, context: Dict[str, float], epsilon: float = 0.1) -> int:
        """
        选择臂 (推荐策略)
        
        Args:
            context: 上下文特征
            epsilon: 探索概率
        """
        # Epsilon-greedy
        if random.random() < epsilon:
            return random.randint(0, self.n_arms - 1)
        
        # 基于上下文调整估值
        adjusted_values = self.values.copy()
        
        for feature, weight in context.items():
            if feature in self.context_weights:
                for i in range(self.n_arms):
                    adjusted_values[i] += self.context_weights[feature][i] * weight
        
        return adjusted_values.index(max(adjusted_values))
    
    def update(self, arm: int, reward: float, context: Dict[str, float]):
        """更新臂的统计"""
        self.counts[arm] += 1
        n = self.counts[arm]
        
        # 更新均值
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward
        
        # 更新上下文权重
        for feature, weight in context.items():
            if feature not in self.context_weights:
                self.context_weights[feature] = [0.0] * self.n_arms
            
            old_weight = self.context_weights[feature][arm]
            self.context_weights[feature][arm] = ((n - 1) / n) * old_weight + (1 / n) * reward * weight


class RecommendationEngine:
    """
    个性化推荐引擎主类
    整合多种推荐算法
    """
    
    def __init__(self, user_profile: Any):
        self.user = user_profile
        
        # 子模块
        self.interest_model = UserInterestModel()
        self.collaborative_filter = CollaborativeFilter()
        self.bandit = ContextualBandit(n_arms=5)
        
        # 推荐历史 (去重)
        self.recommendation_history: set = set()
        
        # 算法权重
        self.algorithm_weights = {
            "content_based": 0.4,
            "collaborative": 0.3,
            "knowledge_based": 0.2,
            "exploration": 0.1
        }
        
    def recommend(self, 
                  content_pool: List[ContentItem],
                  context: Dict = None,
                  n: int = 5) -> List[Recommendation]:
        """
        生成推荐
        
        Args:
            content_pool: 内容池
            context: 当前上下文
            n: 推荐数量
            
        Returns:
            推荐列表
        """
        context = context or {}
        recommendations = []
        
        # 为每个内容计算综合分数
        scored_items = []
        
        for item in content_pool:
            # 跳过已推荐的内容
            if item.id in self.recommendation_history:
                continue
            
            score, reason, confidence = self._calculate_score(item, context)
            scored_items.append((item, score, reason, confidence))
        
        # 排序并选择Top N
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        for item, score, reason, confidence in scored_items[:n]:
            recommendations.append(Recommendation(
                item=item,
                score=score,
                reason=reason,
                confidence=confidence
            ))
            self.recommendation_history.add(item.id)
        
        return recommendations
    
    def _calculate_score(self, item: ContentItem, 
                         context: Dict) -> Tuple[float, str, float]:
        """
        计算内容的综合推荐分数
        
        Returns:
            (分数, 推荐理由, 置信度)
        """
        scores = {}
        
        # 1. 基于内容的推荐分数
        scores["content_based"] = self._content_based_score(item)
        
        # 2. 协同过滤分数
        scores["collaborative"] = self._collaborative_score(item)
        
        # 3. 知识-based分数 (基于用户画像)
        scores["knowledge_based"] = self._knowledge_based_score(item)
        
        # 4. 探索分数 (新颖性)
        scores["exploration"] = self._exploration_score(item)
        
        # 加权综合
        total_score = sum(
            scores[algo] * weight 
            for algo, weight in self.algorithm_weights.items()
        )
        
        # 确定推荐理由
        top_algo = max(scores, key=scores.get)
        reason = self._get_reason(top_algo, item)
        
        # 计算置信度
        confidence = self._calculate_confidence(scores)
        
        return total_score, reason, confidence
    
    def _content_based_score(self, item: ContentItem) -> float:
        """基于内容的推荐分数"""
        score = 0
        
        # 主题匹配
        for topic in item.topics:
            score += self.interest_model.topic_scores[topic]
        
        # 内容类型匹配
        type_score = self.interest_model.content_type_scores[item.content_type]
        score += type_score * 0.5
        
        # 难度匹配
        difficulty_match = self._match_difficulty(item.difficulty)
        score += difficulty_match * 0.3
        
        # 归一化
        return max(0, min(1, (score + 2) / 4))  # 映射到0-1
    
    def _collaborative_score(self, item: ContentItem) -> float:
        """协同过滤分数"""
        predicted = self.collaborative_filter.predict_rating(
            self.user.user_id, item.id
        )
        return predicted
    
    def _knowledge_based_score(self, item: ContentItem) -> float:
        """基于知识的推荐分数"""
        score = 0.5  # 基础分
        
        # 基于Big Five人格调整
        personality = self.user.personality
        
        # 开放性高的人喜欢新颖内容
        if personality.openness > 0.7:
            novelty = self.interest_model.calculate_novelty_score(item)
            score += novelty * 0.2
        
        # 尽责性高的人喜欢结构化内容
        if personality.conscientiousness > 0.7:
            if item.content_type in ["tutorial", "guide", "documentation"]:
                score += 0.2
        
        # 基于认知风格
        cognitive = self.user.cognitive_style
        
        # 视觉型用户偏好有图表的内容
        if cognitive.visual_verbal < 0.3:
            if item.metadata.get("has_visuals", False):
                score += 0.15
        
        return min(1, score)
    
    def _exploration_score(self, item: ContentItem) -> float:
        """探索分数 (新颖性)"""
        novelty = self.interest_model.calculate_novelty_score(item)
        
        # 根据用户开放性调整
        openness = self.user.personality.openness
        exploration_weight = 0.2 + openness * 0.3  # 0.2 - 0.5
        
        return novelty * exploration_weight
    
    def _match_difficulty(self, difficulty: str) -> float:
        """匹配难度偏好"""
        difficulty_levels = {
            "beginner": 1,
            "intermediate": 2,
            "advanced": 3,
            "expert": 4
        }
        
        user_level = difficulty_levels.get(self.user.career.experience_level, 2)
        item_level = difficulty_levels.get(difficulty, 2)
        
        # 偏好略高或同等级
        diff = abs(user_level - item_level)
        if item_level > user_level:
            return 0.5 - diff * 0.1  # 略惩罚过高难度
        else:
            return 0.5 - diff * 0.2  # 更惩罚过低难度
    
    def _get_reason(self, algorithm: str, item: ContentItem) -> str:
        """生成推荐理由"""
        reasons = {
            "content_based": f"基于您对{', '.join(item.topics[:2])}的兴趣",
            "collaborative": "相似用户也喜欢的内容",
            "knowledge_based": "符合您的学习风格",
            "exploration": "探索您可能感兴趣的新领域"
        }
        return reasons.get(algorithm, "个性化推荐")
    
    def _calculate_confidence(self, scores: Dict[str, float]) -> float:
        """计算推荐置信度"""
        # 基于分数一致性计算
        values = list(scores.values())
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        
        # 方差越小，置信度越高
        consistency = 1 - min(1, variance * 4)
        
        # 基于数据量的调整
        data_confidence = min(1, self.user.learning_stats["total_interactions"] / 50)
        
        return (consistency * 0.5 + data_confidence * 0.5)
    
    def record_feedback(self, item_id: str, feedback_type: str, 
                       context: Dict = None):
        """记录反馈"""
        context = context or {}
        
        # 更新兴趣模型
        # 注意：这里需要实际的内容对象，简化处理
        
        # 更新协同过滤
        weight = self.interest_model._get_interaction_weight(feedback_type, None)
        self.collaborative_filter.add_interaction(
            self.user.user_id, item_id, weight
        )
        
        # 更新Bandit
        arm = context.get("arm", 0)
        reward = 1 if feedback_type in ["save", "share", "complete"] else 0
        self.bandit.update(arm, reward, context)
    
    def get_recommendation_diversity(self, recommendations: List[Recommendation]) -> float:
        """计算推荐的多样性"""
        if len(recommendations) < 2:
            return 0.0
        
        # 基于主题覆盖度计算多样性
        all_topics = set()
        for rec in recommendations:
            all_topics.update(rec.item.topics)
        
        # 平均每个内容覆盖的主题数
        avg_topics = sum(len(r.item.topics) for r in recommendations) / len(recommendations)
        
        # 多样性分数
        diversity = len(all_topics) / (avg_topics * len(recommendations))
        
        return min(1, diversity)
    
    def explain_recommendation(self, recommendation: Recommendation) -> str:
        """生成推荐解释"""
        item = recommendation.item
        
        explanation = f"推荐《{item.title}》的原因是：\n"
        explanation += f"• {recommendation.reason}\n"
        
        # 添加个性化细节
        if item.topics:
            top_topics = sorted(
                item.topics,
                key=lambda t: self.interest_model.topic_scores[t],
                reverse=True
            )[:2]
            if top_topics:
                explanation += f"• 涉及您关注的主题: {', '.join(top_topics)}\n"
        
        explanation += f"• 推荐置信度: {recommendation.confidence:.0%}"
        
        return explanation


class SerendipityEngine:
    """
    意外发现引擎
    在保持相关性的同时增加惊喜元素
    """
    
    def __init__(self, interest_model: UserInterestModel):
        self.interest_model = interest_model
        
    def inject_serendipity(self, 
                          base_recommendations: List[Recommendation],
                          content_pool: List[ContentItem],
                          serendipity_ratio: float = 0.2) -> List[Recommendation]:
        """
        向推荐列表注入意外发现内容
        
        Args:
            base_recommendations: 基础推荐
            content_pool: 内容池
            serendipity_ratio: 意外发现内容比例
            
        Returns:
            混合后的推荐列表
        """
        n_serendipity = max(1, int(len(base_recommendations) * serendipity_ratio))
        
        # 筛选意外发现内容
        serendipity_candidates = []
        
        for item in content_pool:
            # 跳过已在基础推荐中的
            if any(r.item.id == item.id for r in base_recommendations):
                continue
            
            # 计算意外发现分数
            score = self._calculate_serendipity_score(item)
            if score > 0.5:
                serendipity_candidates.append((item, score))
        
        # 选择最佳意外发现内容
        serendipity_candidates.sort(key=lambda x: x[1], reverse=True)
        selected = serendipity_candidates[:n_serendipity]
        
        # 创建推荐对象
        serendipity_recs = [
            Recommendation(
                item=item,
                score=score,
                reason="意外发现：可能拓展您的视野",
                confidence=0.6
            )
            for item, score in selected
        ]
        
        # 混合 (间隔插入)
        result = []
        base_idx = 0
        ser_idx = 0
        
        for i in range(len(base_recommendations) + len(serendipity_recs)):
            if i % (int(1/serendipity_ratio)) == 0 and ser_idx < len(serendipity_recs):
                result.append(serendipity_recs[ser_idx])
                ser_idx += 1
            elif base_idx < len(base_recommendations):
                result.append(base_recommendations[base_idx])
                base_idx += 1
        
        return result
    
    def _calculate_serendipity_score(self, item: ContentItem) -> float:
        """
        计算内容的意外发现分数
        
        平衡相关性和新颖性
        """
        # 基础相关性
        relevance = 0
        for topic in item.topics:
            score = self.interest_model.topic_scores[topic]
            relevance += max(0, score)  # 只考虑正相关
        
        relevance /= max(len(item.topics), 1)
        
        # 新颖性
        novelty = self.interest_model.calculate_novelty_score(item)
        
        # 意外发现 = 相关性 * 新颖性 (两者都需要)
        serendipity = relevance * novelty * 4  # 放大
        
        return min(1, serendipity)


# ============================================================================
# 使用示例
# ============================================================================

def example_recommendation():
    """推荐引擎使用示例"""
    
    # 模拟用户档案
    class MockUser:
        def __init__(self):
            self.user_id = "user_001"
            self.personality = type('obj', (object,), {
                'openness': 0.8,
                'conscientiousness': 0.9
            })
            self.career = type('obj', (object,), {
                'experience_level': 'expert'
            })
            self.cognitive_style = type('obj', (object,), {
                'visual_verbal': 0.4
            })
            self.learning_stats = {"total_interactions": 30}
    
    user = MockUser()
    engine = RecommendationEngine(user)
    
    # 模拟内容池
    content_pool = [
        ContentItem(
            id="1",
            title="AI Agent架构设计最佳实践",
            content_type="article",
            topics=["AI", "Agent", "架构"],
            difficulty="advanced",
            estimated_time=15
        ),
        ContentItem(
            id="2",
            title="Python异步编程指南",
            content_type="tutorial",
            topics=["Python", "异步", "编程"],
            difficulty="intermediate",
            estimated_time=20
        ),
        ContentItem(
            id="3",
            title="新兴：量子机器学习入门",
            content_type="article",
            topics=["量子计算", "机器学习", "前沿"],
            difficulty="expert",
            estimated_time=30
        ),
        ContentItem(
            id="4",
            title="团队管理心理学",
            content_type="article",
            topics=["管理", "心理学", "团队"],
            difficulty="intermediate",
            estimated_time=12
        )
    ]
    
    # 模拟历史兴趣
    engine.interest_model.topic_scores["AI"] = 0.8
    engine.interest_model.topic_scores["Agent"] = 0.9
    engine.interest_model.topic_scores["架构"] = 0.7
    engine.interest_model.content_type_scores["article"] = 0.8
    
    # 生成推荐
    recommendations = engine.recommend(content_pool, n=3)
    
    print("=== 个性化推荐 ===\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.item.title}")
        print(f"   分数: {rec.score:.3f}")
        print(f"   原因: {rec.reason}")
        print(f"   置信度: {rec.confidence:.0%}")
        print()
    
    # 添加意外发现
    serendipity = SerendipityEngine(engine.interest_model)
    mixed = serendipity.inject_serendipity(recommendations, content_pool)
    
    print("\n=== 含意外发现的推荐 ===\n")
    for i, rec in enumerate(mixed, 1):
        marker = " [意外发现]" if "意外发现" in rec.reason else ""
        print(f"{i}. {rec.item.title}{marker}")


if __name__ == "__main__":
    example_recommendation()
