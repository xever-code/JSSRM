import os

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.ensemble import GradientBoostingClassifier

# 初始化 session_state 中的 data
# 创建一个空的DataFrame来存储预测数据
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(
        columns=['GGT', 'HDL', 'TP', 'ALB', 'abdominal pain','Prediction','Label'])

# 在主页面上显示数据
st.header('胆囊癌早期诊断模型--GradientBoost')

# st.markdown("### 本地图片示例")
# 创建两列布局
left_column, col1, col2, col3, right_column = st.columns(5)

# 在左侧列中添加其他内容
left_column.write("")

# 在右侧列中显示图像
dirs = os.getcwd()

# 在右侧列中显示图像
right_column.image(dirs + '\\logo.png', caption='', width=100)

# with open("F:\\model\\jssrm\logo2.png", "rb") as f:
#    local_image = f.read()
#    st.image(local_image, caption='', width=100)


# 创建一个侧边栏
st.sidebar.header('输入参数')

# Input bar 1
a = st.sidebar.number_input('GGT', min_value=0.0, max_value=150.0, value=20.1)
b = st.sidebar.number_input('HDL', min_value=0.0, max_value=200.0, value=1.31)
c = st.sidebar.number_input('TP', min_value=0.0, max_value=200.0, value=72.6)
d = st.sidebar.number_input('ALB', min_value=0.0, max_value=1500.0, value=44.0)
e = st.sidebar.number_input('abdominal pain', min_value=0, max_value=1, value=0)

# Unpickle classifier
mm = joblib.load('E:\\Py\\JSSRM\\GradientBoost.pkl')

# If button is pressed
if st.sidebar.button("Submit"):
    # Store inputs into dataframe
    X = pd.DataFrame([[a, b, c, d, e]],
                     columns=['GGT', 'HDL', 'TP', 'ALB', 'abdominal pain'])
    # X = X.replace(["Brown", "Blue"], [1, 0])

    # Get prediction
    for index, row in X.iterrows():
        data1 = row.to_frame()
        data2 = pd.DataFrame(data1.values.T, columns=data1.index)
        result111 = mm.predict(data2)
        result222 = str(result111).replace("[", "")
        result = str(result222).replace("]", "")  # 预测结果
        result333 = mm.predict_proba(data2)
        result444 = str(result333).replace("[[", "")
        result555 = str(result444).replace("]]", "")
        strlist = result555.split(' ')
        # print(index,row['OUTPATIENT_ID'],result)
        result_prob_neg = round(float(strlist[0]) * 100, 2)
        if len(strlist[1]) == 0:
            result_prob_pos = 'The conditions do not match and cannot be predicted'
        else:
            result_prob_pos = round(float(strlist[1]) * 100, 2)  # 预测概率
    explainer = shap.TreeExplainer(mm)
    shap_values = explainer.shap_values(data2)
    shap_values = shap_values.reshape((1, -1))
    # output_index = 0  # 假设我们选择第一个输出来解释
    # 绘制 SHAP 分析图
    # 构建特征向量
    # features = np.array([a, b, c, d, e,f,g])  # 假设只有 7 个参数
    # shap.force_plot(explainer.expected_value[0], shap_values[0], data2.iloc[0])
    # st.pyplot()

    # Output prediction
    st.text(f"The probability of GradientBoost is: {str(result_prob_pos)}%")
    # st.text({str(shap_values[0])})

    # 创建一个新的DataFrame来存储用户输入的数据
    new_data = pd.DataFrame([[a, b, c, d, e, result_prob_pos / 100, None]],
                            columns=st.session_state['data'].columns)

    # 将预测结果添加到新数据中
    st.session_state['data'] = pd.concat([st.session_state['data'], new_data], ignore_index=True)

# 上传文件按钮
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # 读取 Excel 文件
    df = pd.read_excel(uploaded_file)

    # 列名映射字典,左为Excel字段，右为模型参数名
    column_mapping = {
        'GGT': 'GGT',
        'HDL': 'HDL',
        'TP': 'TP',
        'ALB': 'ALB',
        'abdominal pain': 'abdominal pain'
    }

    # 假设 'Label' 列在 Excel 文件中存在并且不参与计算
    label_column = 'label'  # 这是 Excel 文件中未参与计算的列名

    # 进行列名映射
    df = df.rename(columns=column_mapping)

    # 检查是否所有必需的列都存在
    missing_cols = [col for col in ['GGT', 'HDL', 'TP', 'ALB', 'abdominal pain'] if
                    col not in df.columns]

    if missing_cols:
        st.error(f"Missing columns in the uploaded file: {', '.join(missing_cols)}")
    else:
        # 逐行读取数据并进行预测
        for _, row in df.iterrows():
            # 提取每一行数据并转换为模型输入格式
            X = pd.DataFrame([row],
                             columns=['GGT', 'HDL', 'TP', 'ALB', 'abdominal pain'])

            # 进行预测
            result = mm.predict(X)[0]
            result_prob = mm.predict_proba(X)[0][1]

            # 获取标签列的值
            label = row[label_column] if label_column in row else None

            # 将结果添加到 session_state 的 data 中
            new_data = pd.DataFrame([[row["GGT"], row["HDL"], row["TP"], row["ALB"], row["abdominal pain"], result_prob, label]],
                                    columns=st.session_state['data'].columns)
            st.session_state['data'] = pd.concat([st.session_state['data'], new_data], ignore_index=True)

# 显示更新后的 data
st.write(st.session_state['data'])

# Footer
st.write(
    "<p style='font-size: 12px;'>Disclaimer: This mini app is designed to provide general information and is not a substitute for professional medical advice or diagnosis. Always consult with a qualified healthcare professional if you have any concerns about your health.</p>",
    unsafe_allow_html=True)
st.markdown('<div style="font-size: 12px; text-align: right;">Powered by MyLab+ i-Research Consulting Team</div>',
            unsafe_allow_html=True)
