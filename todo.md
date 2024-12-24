1. check path bug   DONE
2. draw graph       DONE    
3. check cache_hit_rate     DONE
4. check score method       DONE
5. check cache data bug
6. check if async in multi extraction keywords  
修改top_k获取，直接从所有数据中取前k个

考虑一个chunk提取出几个实体、关系

---index---
修改content嵌入模板       done
每个节点构建连接自己的一条边     done   
图构建完成后，校验、去除冗余（做节点和边的筛选，desciption更新方式(做summary)）done 
index中定义history属性存储所有description的拼接   done    ---是否删除history？

---query---
query的时候修改prompt让entity、relation生成对应的description(relation额外生成src,tgt)拼接在一起做嵌入
更新游走方式



去重在chosen_edge还是edge_candidate
index完那一部分效率加快

测试naive和local有无bug