import pandas as pd

def time_format(T):
    """
    :param T:各种时间格式混合在一起
    :return: format time
    """
    new_T=pd.to_datetime(T)
    time_formatted =[]
    for new_t in new_T:
        new=new_t.replace(year=new_t.year -100 if new_t.year > 2015 else new_t.year)
        time_formatted.append(new)
    return time_formatted


def main():
    ori_data=pd.read_csv("D:\应用程序\PythonProject\learn\Data\movies.csv")

    # imdb_id和homepage对于分析的帮助较小，可以考虑直接删掉，其余有用字段的空值由"No data"填充
    ori_data.drop(['imdb_id','homepage'],axis=1,inplace=True)
    data = ori_data.fillna("No data")
    # print(pd.isnull(data).sum().sort_values(ascending=False))
    # 统一时间格式
    data['release_date']=time_format(data['release_date'])
    print(data.groupby('director')['popularity'].agg('mean').sort_values(ascending=False).reset_index())





if __name__=="__main__":
    main()
