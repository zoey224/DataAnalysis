```python
import numpy as np
import pandas as pd
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randint(10, size=(6, 4)), index=dates, columns=list('ABCD'))
```

df的结果是：

```python
Int[47]: df
Out[47]: 
            A  B  C  D
2013-01-01  1  4  6  2
2013-01-02  7  2  9  2
2013-01-03  4  1  2  9
2013-01-04  1  3  4  7
2013-01-05  9  2  4  8
2013-01-06  0  8  1  9

Int[47]:type(df)
Out[48]: pandas.core.frameData.DataFrame
```

```python
Int[49]:df.to_numpy()
Out[49]: 
array([[1, 4, 6, 2],
       [7, 2, 9, 2],
       [4, 1, 2, 9],
       [1, 3, 4, 7],
       [9, 2, 4, 8],
       [0, 8, 1, 9]])

Int[50]:type(df.to_numpy())
Out[50]: numpy.ndarray
```

numpy对应ndarray的数据类型 ，pandas对应DataFrame的数据类型 。 从上文可以看出，通过代码pd.DataFrame(np.random.randint(10, size=(6, 4)), index=dates, columns=list('ABCD'))  给ndarray添加索引(index)信息和字段信息(column)从而形成DataFrame。所以可以这样认为：在python数据分析领域，pandas是对numpy的补充而不是替代。在实际应用场景中，A，B，C，D就是真是存在且具有意义的一个字段（在机器学习领域，字段通常被称之为特征，有时候也被称为维度，即多个特征就是代表多个维度），比如在电影数据集中，A，B，C，D字段可能就代表电影的名字，电影的导演，电影的票房以及电影的种类等等。

给单纯数据集添加字段信息的好处就是：便于对结构化的数据进行数据分析。接下来举一个简单的例子：如果字段D表示电影的票房，那么大家一般都会对票房排行感兴趣，此时就可以直接对字段D中的数值进行排序，而不用关心其他的字段。

```python

In [54]:df.sort_values(by='D',ascending=False)
Out[54]: 
            A  B  C  D
2013-01-03  4  1  2  9
2013-01-06  0  8  1  9
2013-01-05  9  2  4  8
2013-01-04  1  3  4  7
2013-01-01  1  4  6  2
2013-01-02  7  2  9  2
```

接下来为了更好的进行数据分析，模拟一个小型的电影数据集：数据集一共有10条记录，五个字段，其中0.0值表示异常值，NaN值表示缺失值。五个字段所代表的含义分别为：original_title ：电影名字 |  director : 导演  | cast : 主演
genres : 类型   | revenue : 票房

```python
Int[1]: df_movies=pd.DataFrame({
'original_title':['战狼','西虹市首富','红海行动','流浪地球','海王','美人鱼','前任3','飞驰人生','羞羞的铁拳','变形金刚4','头号玩家'],
'director':['吴京','闫非','林超贤','郭帆','温子仁','周星驰','田羽生','韩寒','宋阳',np.NaN,'斯皮尔伯格'],
'cast':['吴京','沈腾','张译','吴京',np.NaN,'邓超','韩庚','沈腾','沈腾',np.NaN,'佩恩'],
'genres':['战争','喜剧','战争','科幻','动作','喜剧','爱情','喜剧','喜剧','科幻','科幻'],
'revenue':[56.8,25.3,30.2,46.18,0.0,33.9,19.26,17.03,21.9,19.79,0.0]})
Out[1] :
        original_title director cast    genres revenue
0             战狼       吴京    吴京     战争    56.8
1          西虹市首富     闫非   沈腾     喜剧    25.3
2           红海行动     林超贤  张译     战争    30.2
3           流浪地球      郭帆   吴京     科幻   46.18
4             海王      温子仁   NaN     动作     0.0
5            美人鱼     周星驰   邓超     喜剧    33.9
6            前任3      田羽生   韩庚     爱情   19.26
7           飞驰人生     韩寒    沈腾     喜剧   17.03
8          羞羞的铁拳    宋阳    沈腾     喜剧    21.9
9          变形金刚4      Nan    NaN     科幻   19.79
10         头号玩家  斯皮尔伯格  佩恩     科幻    0.00
In [2]:df_movies.dtypes
Out[2]: 
original_title     object
director           object
cast               object
genres             object
revenue           float64
dtype: object
```

在获得数据集以后，一般需要对数据集的质量进行分析。通常情况是分析数据集中哪几个字段存在缺失值，异常值以及这这些值的个数（占比）。

```python
# 检查NaN值的分布情况
In [7]:pd.isna(df_movies)
Out[7]: 
   original_title  director   cast  genres  revenue
0           False     False  False   False    False
1           False     False  False   False    False
2           False     False  False   False    False
3           False     False  False   False    False
4           False     False   True   False    False
5           False     False  False   False    False
6           False     False  False   False    False
7           False     False  False   False    False
8           False     False  False   False    False
9           False      True   True   False    False
10          False     False  False   False    False
#直观了解哪些字段存在NaN值
In [8]:pd.isna(df_movies).any()
Out[8]: 
original_title    False
director           True
cast               True
genres            False
revenue           False
dtype: bool
#了解NaN值的个数来判断数据集的质量   
In [9]:pd.isna(df_movies).sum()
Out[9]: 
original_title    0
director          1
cast              2
genres            0
revenue           0
dtype: int64
#pandas中另一种一维的数据格式Series
In [10]:type(pd.isna(df_movies).sum())
Out[10]: pandas.core.series.Series
#对每个字段含有NaN值个数进行倒序排序，来辅助后续的数据清洗工作。
#ascending 上升的，也就是从低到高正序排序
In [11]:pd.isna(df_movies).sum().sort_values(ascending=False)
Out[11]: 
cast              2
director          1
revenue           0
genres            0
original_title    0
dtype: int64 
```

对缺失值和异常值进行处理，由于在本数据集中缺失值较少而不影响整体的数据分析，所以可以直接将缺失值去除，而在实际很多项目中往往对缺失值进行填充。

```python
#去除缺失值
moviesdata_dropna=df_movies.dropna()
moviesdata_dropna
Out[34]: 
    original_title director cast genres     revenue
0      战狼          吴京    吴京     战争    56.80
1    西虹市首富       闫非    沈腾     喜剧    25.30
2      红海行动      林超贤   张译     战争    30.20
3      流浪地球       郭帆    吴京     科幻    46.18
5      美人鱼        周星驰   邓超     喜剧    33.90
6      前任3         田羽生   韩庚     爱情    19.26
7    飞驰人生         韩寒    沈腾     喜剧    17.03
8    羞羞的铁拳      宋阳     沈腾     喜剧    21.90
10    头号玩家     斯皮尔伯格  佩恩     科幻     0.00
#去除异常值
moviesdata_cleaned=moviesdata_dropna[moviesdata_dropna['revenue']>0]
moviesdata_cleaned
Out[39]: 
   original_title director cast genres     revenue
0        战狼        吴京    吴京     战争    56.80
1    西虹市首富       闫非    沈腾     喜剧    25.30
2      红海行动      林超贤   张译     战争    30.20
3      流浪地球       郭帆    吴京     科幻    46.18
5      美人鱼        周星驰   邓超     喜剧    33.90
6      前任3         田羽生   韩庚     爱情    19.26
7     飞驰人生         韩寒    沈腾     喜剧    17.03
8    羞羞的铁拳       宋阳    沈腾     喜剧    21.90

moviesdata_cleaned.shape
Out[41]: (8, 5)

moviesdata_dropna['revenue']>0
Out[43]: 
0      True
1      True
2      True
3      True
5      True
6      True
7      True
8      True
10    False
Name: revenue, dtype: bool

type(moviesdata_dropna['revenue']>0)
Out[44]: pandas.core.series.Series
```

对数据进行清洗之后，需要对数据进行分析。

```python
#按照票房进行从高到低排序
moviesdata_cleaned.sort_values('revenue',ascending=False)
Out[47]: 
          original_title director cast   genres  revenue
0             战狼        吴京   吴京     战争    56.80
3           流浪地球       郭帆   吴京     科幻    46.18
5            美人鱼      周星驰   邓超     喜剧    33.90
2           红海行动     林超贤   张译     战争    30.20
1          西虹市首富      闫非   沈腾     喜剧    25.30
8          羞羞的铁拳      宋阳   沈腾     喜剧    21.90
6            前任3       田羽生   韩庚     爱情    19.26
7           飞驰人生       韩寒   沈腾     喜剧    17.03
#有时候，人们只关心哪部电影多少票房，不会关心其他的字段信息，所以此时只需要显示original_title和revenue字段信息。
#在思想上类似于sql中的查询操作：
select original_title,revenue from moviesdata_cleaned order by revenue desc
#在实际操作中类似于Numpy中的数据切分:
#以下步骤是 展示Numpy数组的切分过程
moviesdata_cleaned.sort_values('revenue',ascending=False).to_numpy()#将DataFrame转化为ndarray
Out[53]: 
array([['战狼', '吴京', '吴京', '战争', 56.8],
       ['流浪地球', '郭帆', '吴京', '科幻', 46.18],
       ['美人鱼', '周星驰', '邓超', '喜剧', 33.9],
       ['红海行动', '林超贤', '张译', '战争', 30.2],
       ['西虹市首富', '闫非', '沈腾', '喜剧', 25.3],
       ['羞羞的铁拳', '宋阳', '沈腾', '喜剧', 21.9],
       ['前任3', '田羽生', '韩庚', '爱情', 19.26],
       ['飞驰人生', '韩寒', '沈腾', '喜剧', 17.03]], dtype=object)
#[:,[0,4]]根据位置信息取第一列和第五列的所有数据
moviesdata_cleaned.sort_values('revenue',ascending=False).to_numpy()[:,[0,4]]
Out[58]: 
array([['战狼', 56.8],
       ['流浪地球', 46.18],
       ['美人鱼', 33.9],
       ['红海行动', 30.2],
       ['西虹市首富', 25.3],
       ['羞羞的铁拳', 21.9],
       ['前任3', 19.26],
       ['飞驰人生', 17.03]], dtype=object)
#numpy中元素也可以有不同的数据类型
moviesdata_cleaned.sort_values('revenue',ascending=False).to_numpy()[0,4]
Out[54]: 56.8
type(moviesdata_cleaned.sort_values('revenue',ascending=False).to_numpy()[0,4])
Out[55]: float
   
#不同于ndarray的是，DataFrame在列方向可以直接根据字段的名字来进行切分操作。
moviesdata_cleaned.sort_values('revenue',ascending=False)[['original_title','revenue']]
Out[62]: 
        original_title  revenue
0             战狼       56.80
3           流浪地球     46.18
5            美人鱼      33.90
2           红海行动     30.20
1          西虹市首富    25.30
8          羞羞的铁拳    21.90
6            前任3      19.26
7           飞驰人生    17.03
#在列方向也可以根据位置信息进行切分操作 
moviesdata_cleaned.sort_values('revenue',ascending=False).iloc[:,[0,4]]
Out[72]: 
        original_title  revenue
0             战狼       56.80
3           流浪地球     46.18
5            美人鱼      33.90
2           红海行动     30.20
1          西虹市首富    25.30
8          羞羞的铁拳    21.90
6            前任3       19.26
7           飞驰人生     17.03

#有需要注意的地方
moviesdata_cleaned.sort_values('revenue',ascending=False)['revenue']
Out[60]: 
0    56.80
3    46.18
5    33.90
2    30.20
1    25.30
8    21.90
6    19.26
7    17.03
Name: revenue, dtype: float64
type(moviesdata_cleaned.sort_values('revenue',ascending=False)['revenue'])
Out[64]: pandas.core.series.Series#数据类型为series
    
moviesdata_cleaned.sort_values('revenue',ascending=False)[['revenue']]
Out[63]: 
   revenue
0    56.80
3    46.18
5    33.90
2    30.20
1    25.30
8    21.90
6    19.26
7    17.03
type(moviesdata_cleaned.sort_values('revenue',ascending=False)[['revenue']])
Out[65]: pandas.core.frame.DataFrame#数据类型为DataFrame
```

真实的数据集可能有上万条记录，在对这些记录按照某一个字段排序（此处按照票房排序）之后，有时候不需要知道所有的排序结果，只需要知道票房最高的N条记录，这在Sql中称为求取topN。

```sql
#在Sql中可以通过关键字limit来求取topN：
select original,revenue from moviesdata_cleaned order by revenue desc limit 3;
```

```python
#在pandas中为了求取票房topN,在排序后只取前N行记录就可以完成topN的求取。分析结果如下所示：
        original_title  revenue
0             战狼       56.80
3           流浪地球     46.18
5            美人鱼      33.90
#利用pandas可以有多种不同的方案得出上面的结果，这也是python这门语言灵活的地方。
moviesdata_cleaned.sort_values('revenue',ascending=False)[['original_title', 'revenue']][0:3]
Out[84]: 
       original_title  revenue
0             战狼      56.80
3           流浪地球    46.18
5            美人鱼     33.90

moviesdata_cleaned.sort_values('revenue',ascending=False)[0:3][['original_title', 'revenue']]
Out[85]: 
     original_title  revenue
0             战狼      56.80
3           流浪地球    46.18
5            美人鱼     33.90
#需要注意的地方：通过以下代码块可以得知loc在行方向是根据index值切取数据，在求取topN时得不到想要的结果，因为按照票房排序后index值变得无序，而iloc则是根据行的位置信息切分，可以顺利得到结果。但是不建议通过iloc[:3,[0, 4]]求取top3，因为这首先需要明确需表达字段所在列的位置。在字段较多的数据集以及分析任务很多的项目中，这种方式很影响数据分析的效率。
moviesdata_cleaned.sort_values('revenue',ascending=False).loc[:3,['original_title', 'revenue']]
Out[87]: 
      original_title  revenue
0             战狼     56.80
3           流浪地球    46.18

moviesdata_cleaned.sort_values('revenue',ascending=False).iloc[:3,[0, 4]]
Out[91]: 
      original_title  revenue
0             战狼    56.80
3           流浪地球    46.18
5            美人鱼    33.90   
```

Pandas的强大之处除了通过字段名称(有时候也可以是index值)和行列位置获取你想要的数据信息外，也可以通过布尔索引(Boolean indexing)来获取数据集中的信息。接下来通过实际数据分析中的场景，来阐述布尔索引的强大之处。在电影票房的分析中，人们除了关注票房排行版之外，还比较关注票房超过某一值的电影有哪些，比如票房超过30亿的电影有哪些，类似的场景在其他项目中也经常遇见。

```python
#先初步了解布尔索引是如何工作的
#使用与列长度相同(即记录条数)的Boolean类型(Series,list,ndarray)进行Boolean索引
df
Out[106]: 
   A  B  C  D
0  1  4  6  2
1  7  2  9  2
2  4  1  2  9
3  1  3  4  7
4  9  2  4  8
5  0  8  1  9
#Series
df['A']>5
Out[109]: 
0    False
1     True
2    False
3    False
4     True
5    False
Name: A, dtype: bool 
df[df['A']>5]
Out[110]: 
   A  B  C  D
1  7  2  9  2
4  9  2  4  8
#list
df[[False,True,False,False,True,False]]
Out[111]: 
   A  B  C  D
1  7  2  9  2
4  9  2  4  8
#ndarray
df[np.array([False,True,False,False,True,False])]
Out[112]: 
   A  B  C  D
1  7  2  9  2
4  9  2  4  8
#需要注意的是：使用只有一列的Boolean类型DataFrame做布尔索引切分操作时，
df[['C']]>5
Out[116]: 
       C
0   True
1   True
2  False
3  False
4  False
5  False
df[df[['C']]>5]
Out[117]: 
    A   B    C   D
0 NaN NaN  6.0 NaN
1 NaN NaN  9.0 NaN
2 NaN NaN  NaN NaN
3 NaN NaN  NaN NaN
4 NaN NaN  NaN NaN
5 NaN NaN  NaN NaN
df[['A']]>5
Out[119]: 
       A
0  False
1   True
2  False
3  False
4   True
5  False
Out[120]: 
     A   B   C   D
0  NaN NaN NaN NaN
1  7.0 NaN NaN NaN
2  NaN NaN NaN NaN
3  NaN NaN NaN NaN
4  9.0 NaN NaN NaN
5  NaN NaN NaN NaN
#
bool_pd=pd.DataFrame([True,False,True,False,True,False],columns=['E'])

bool_pd
Out[125]: 
       E
0   True
1  False
2   True
3  False
4   True
5  False

df[bool_pd]
Out[126]: 
    A   B   C   D
0 NaN NaN NaN NaN
1 NaN NaN NaN NaN
2 NaN NaN NaN NaN
3 NaN NaN NaN NaN
4 NaN NaN NaN NaN
5 NaN NaN NaN NaN

#针对分类数据的选取操作。
```

接下来通过布尔索引就可以轻松的获取电影票房高于30亿的电影；

```python
#布尔索引通过moviesdata_cleaned['revenue']>30实现 ，其返回一个布尔Series
moviesdata_cleaned[moviesdata_cleaned['revenue']>30]
Out[13]: 
          original_title director cast   genres  revenue
0             战狼         吴京   吴京      战争    56.80
2           红海行动      林超贤   张译     战争    30.20
3           流浪地球       郭帆   吴京      科幻    46.18
5            美人鱼      周星驰   邓超      喜剧    33.90
```

除了moviesdata_cleaned['revenue']>30这样的场景，往往还有类似30<moviesdata_cleaned['revenue']<40的场景；

```python
#通过(moviesdata_cleaned['revenue']<40) & (moviesdata_cleaned['revenue']>30)进行索引
moviesdata_cleaned[(moviesdata_cleaned['revenue']<40) & (moviesdata_cleaned['revenue']>30)]
Out[24]: 
  original_title director cast  genres  revenue
2     红海行动    林超贤   张译     战争     30.2
5     美人鱼      周星驰   邓超     喜剧     33.9
##只有(moviesdata_cleaned['revenue']<40) & (moviesdata_cleaned['revenue']>30)满足需求
(moviesdata_cleaned['revenue']<40) & (moviesdata_cleaned['revenue']>30)
Out[25]: 
0    False
1    False
2     True
3    False
5     True
6    False
7    False
8    False
Name: revenue, dtype: bool

Int[26]:moviesdata_cleaned['revenue']<40 & moviesdata_cleaned['revenue']>30
TypeError: cannot compare a dtyped [float64] array with a scalar of type [bool]
    
(moviesdata_cleaned['revenue']<40) & moviesdata_cleaned['revenue']>30
Out[27]: 
0    False
1    False
2    False
3    False
5    False
6    False
7    False
8    False
Name: revenue, dtype: bool
        
moviesdata_cleaned['revenue']<40 & (moviesdata_cleaned['revenue']>30)
Out[28]: 
0    False
1    False
2    False
3    False
5    False
6    False
7    False
8    False
Name: revenue, dtype: bool
```

以上都是对数值型字段进行操作，在很多实际场景中，经常 需要对分类字段进行相关操作。比如需要知道某一种类的电影有哪些。

```python 
#需求：战争和科幻电影有哪些部 
moviesdata_cleaned['genres'].isin(['战争','科幻'])
Out[29]: 
0     True
1    False
2     True
3     True
5    False
6    False
7    False
8    False
Name: genres, dtype: bool
        
moviesdata_cleaned[moviesdata_cleaned['genres'].isin(['战争','科幻'])]
Out[30]: 
         original_title director cast genres  revenue
0             战狼       吴京   吴京     战争    56.80
2           红海行动     林超贤  张译     战争    30.20
3           流浪地球      郭帆   吴京     科幻    46.18
```

给以上数据集添加一个评分字段，便于后续的数据分析。

```python
#添加评分字段
moviesdata_cleaned['score']=[9.0,8.0,8.7,8.8,8.5,7.0,8.3,8.6]
moviesdata_cleaned
Out[33]: 
        original_title director cast    genres   revenue score
0             战狼       吴京     吴京     战争    56.80   9.0
1          西虹市首富     闫非     沈腾     喜剧    25.30   8.0
2           红海行动      林超贤   张译     战争    30.20   8.7
3           流浪地球      郭帆     吴京     科幻    46.18   8.8
5            美人鱼      周星驰   邓超     喜剧    33.90   8.5
6            前任3       田羽生   韩庚     爱情    19.26   7.0
7           飞驰人生      韩寒    沈腾     喜剧    17.03   8.3
8          羞羞的铁拳     宋阳    沈腾     喜剧    21.90   8.6

#通过添加一个评分字段后，数据集中有两个数值型数据
#需求：求评分大于30，评分高于8.5的电影
#方案1
moviesdata_cleaned[moviesdata_cleaned['revenue']>30][moviesdata_cleaned['score']>8.5]
__main__:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
Out[39]: 
         original_title director cast   genres  revenue  score
0             战狼       吴京    吴京     战争    56.80    9.0
2           红海行动     林超贤   张译     战争    30.20    8.7
3           流浪地球      郭帆    吴京     科幻    46.18    8.8
#方案2
moviesdata_cleaned[(moviesdata_cleaned['revenue']>30)&(moviesdata_cleaned['score']>8.5)]
Out[41]: 
         original_title director cast   genres  revenue  score
0             战狼       吴京     吴京     战争    56.80    9.0
2           红海行动     林超贤   张译     战争    30.20    8.7
3           流浪地球      郭帆   吴京      科幻    46.18    8.8 
```

对于分类字段还有一个经常出现的场景就是进行分组操作，对数据进行分组之后通常进行聚合操作。聚合操作包括：求和，最大值，最小值，平均值等。

```python
#分组求和
moviesdata_cleaned.groupby('genres').sum()
Out[44]: 
         revenue  score
genres#分组字段                
喜剧        98.13   33.4
战争        87.00   17.7
爱情        19.26    7.0
科幻        46.18    8.8

#分组求平均
 moviesdata_cleaned.groupby('genres').mean()
Out[45]: 
        revenue  score
genres#分组字段                
喜剧      24.5325   8.35
战争      43.5000   8.85
爱情      19.2600   7.00
科幻      46.1800   8.80


#每种类型的电影有多少部
moviesdata_cleaned.groupby('genres').count()
Out[48]: 
        original_title  director  cast  revenue  score
genres                                                
喜剧                   4         4     4        4      4
战争                   2         2     2        2      2
爱情                   1         1     1        1      1
科幻                   1         1     1        1      1
#如果有两个分组字段
moviesdata_cleaned.groupby(['genres','cast']).sum()

Out[49]: 
               revenue  score
genres cast                
喜剧     沈腾      64.23   24.9
        邓超      33.90    8.5
战争     吴京      56.80    9.0
         张译      30.20    8.7
爱情     韩庚      19.26    7.0
科幻     吴京      46.18    8.8
```



























































