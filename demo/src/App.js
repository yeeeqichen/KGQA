import React, { Component } from 'react'
import { Layout, Menu, Breadcrumb, Empty, Collapse, Alert, List, Divider, Typography, Descriptions, Badge } from 'antd';
import Input from './SearchInput'
import * as echarts from 'echarts'
import AnswerList from './AnswerList'
import { UserOutlined, LaptopOutlined, NotificationOutlined } from '@ant-design/icons';
import './App.css'
import $ from 'jquery'
const { SubMenu } = Menu;
const { Header, Content, Sider } = Layout;
const { Panel } = Collapse

function updateGraph(id, data, links, options) {
  let chartDom = document.getElementById(id);
  let myChart = echarts.init(chartDom);
  let option;
  option = {
    title: {
      text: options.title,
      subtext: options.subtext,
    },
    tooltip: {},
    animationDuration: 1500,
    animationEasingUpdate: 'quinticInOut',
    series: [
      {
        name: options.seriesName,
        type: 'graph',
        layout: options.layout,
        data: data,
        links: links,
        roam: 'move',
        zoom: options.zoom,
        emphasis: {
          focus: 'adjacency',
          lineStyle: {
            width: 10
          }
        },
        label: {
          show: true,
          position: 'right',
          formatter: "{b}"
        },
      }
    ]
  };
  myChart.setOption(option);
  option && myChart.setOption(option);
}

function preprocess_subgraph(sub_graph){
  sub_graph = [...sub_graph[0], ...sub_graph[1]]
  let entities = []
  let relations = []
  for (let triple of sub_graph) {
    entities.push(triple[0])
    entities.push(triple[2])
    relations.push(triple[1])
  }
  entities = [...new Set(entities)]
  relations = [...new Set(relations)]
  let dict = {}
  let data = entities.map((value, index) => {
    dict[value] = index
    return {
      id: index,
      name: value,
      draggable: true,
    }
  })
  let links = sub_graph.map((triple) => {
    return {
      source: dict[triple[0]],
      target: dict[triple[2]],
      name: triple[1],
      label: {
        show: true,
        formatter: (params) => {
          return params.data.name
        }
      },
    }
  })
  return { data, links }
}


export default class App extends Component {

  state = {answer: [], data: [], links: [], curOption: 'demo'}

  onSearch = (question) => {
    let _this = this
    $.ajax({
      url: 'http://localhost:2333/predict',
      data: {
        question:question
      },
      success: (res) => {
        res = JSON.parse(res)
        let status = res.status
        // console.log(status)
        if (status === '500') {
          _this.setState({
            status: '500',
            question: question,
          })
          return
        }
        let response_data = res.data
        // console.log(response_data)
        let { answer, sub_graph, head, answer_subgraph } = response_data
        let { data, links } = preprocess_subgraph(sub_graph)
        let answer_graphs = answer_subgraph.map(item => {
          return preprocess_subgraph(item)
        })
        // console.log(data, links, answer, head, answer_graphs)
        _this.setState({
          answer: answer,
          data: data,
          links: links,
          head: head,
          answer_graphs: answer_graphs,
          question: question,
          status: '200 OK',
        })
        updateGraph('main', this.state.data, this.state.links, {
          title: '问题子图',
          subtext: '',
          zoom: 5.5,
          layout: 'force',
          seriesName: '问题子图'
        })

        let answer_options = {
          title: '答案子图',
          subtext: '',
          zoom: 1,
          layout: 'circular',
          seriesName: '答案子图'
        }
        updateGraph('answer1', this.state.answer_graphs[0].data, this.state.answer_graphs[0].links, answer_options)
        updateGraph('answer2', this.state.answer_graphs[1].data, this.state.answer_graphs[1].links, answer_options)
        updateGraph('answer3', this.state.answer_graphs[2].data, this.state.answer_graphs[2].links, answer_options)
      }
    })
  }

  render() {
    let _this = this
    return (
      <Layout>
        <Header className="header">
          <div className="logo" />
          <Menu theme="dark" mode="horizontal" defaultSelectedKeys={['2']}>
            <div>基于知识图谱的问答系统Demo</div>
          </Menu>
        </Header>
        <Layout>
          <Sider width={200} className="site-layout-background">
            <Menu
              mode="inline"
              defaultSelectedKeys={['1']}
              defaultOpenKeys={['sub1']}
              style={{ height: '100%', borderRight: 0 }}
              onClick={(obj) => {
                if (obj.key === '1') {
                  _this.setState({
                    curOption: 'demo',
                    question: undefined,
                  })
                }
                else if (obj.key === '2') {
                  _this.setState({
                    curOption: 'description'
                  })
                }
                else if (obj.key === '3') {
                  _this.setState({
                    curOption: 'aboutUs'
                  })
                }
              }}
            >
              <SubMenu key="sub1" icon={<UserOutlined />} title="系统菜单">
                <Menu.Item key="1">系统功能演示</Menu.Item>
                <Menu.Item key="2">系统介绍说明</Menu.Item>
                <Menu.Item key="3">关于我们</Menu.Item>
              </SubMenu>
            </Menu>
          </Sider>
          <Layout style={{ padding: '0 24px 24px' }}>
            <Breadcrumb style={{ margin: '16px 0' }}>
              <Breadcrumb.Item>{this.state.curOption}</Breadcrumb.Item>
            </Breadcrumb>
            <Content
              className="site-layout-background"
              style={{
                padding: 24,
                margin: 0,
                minHeight: 280,
                width: 750,
              }}
            >
              {
                (() => {
                  if (this.state.curOption === 'demo') {
                    return (
                      <div>
                        <Input onSearch={this.onSearch} />
                        {
                          this.state.question === undefined ?
                            <Empty className="empty" />
                            :
                            (
                              this.state.status === '500' ?
                                <Alert
                                  message="非法的问题内容"
                                  description="无法在知识图谱中获得能够支持问题回答的信息"
                                  type="error"
                                  className="alert"
                                />
                                :
                                <div className="wrapper">
                                  <h3 className="head">问题信息对应子图 <span className="label label-success tag">{this.state.head}</span></h3>
                                  <div><div id='main'></div></div>

                                  <AnswerList answers={this.state.answer} />
                                </div>
                            )
                        }
                      </div>
                    )
                  }
                  else if (this.state.curOption === 'description') {
                    let data = [
                      {
                        name: 'neo4j',
                        usage: '使用neo4j图数据库来存储知识图谱，并对其进行查询来获得相应子图'
                      },
                      {
                        name: 'echarts',
                        usage: '使用echarts来对知识图谱进行可视化'
                      },
                      {
                        name: 'NER',
                        usage: '使用NER系统来对输入的问题进行预处理，获取问题中与知识图谱有关的命名实体信息'
                      }
                    ]
                    return (
                      <div>
                        <Divider orientation="left">问答系统流程示意图</Divider>
                        <img className="descriptionImg" src="/imgs/ModelDiagram.png"></img>
                        <Divider orientation="left">项目demo中额外涉及的相关技术</Divider>
                        <List
                          bordered
                          dataSource={data}
                          renderItem={item => (
                            <List.Item>
                              <Typography.Text mark>[{item.name}]</Typography.Text> {item.usage}
                            </List.Item>
                          )}
                        />
                      </div>
                    )
                  }
                  else if (this.state.curOption === 'aboutUs') {
                    return (
                      <Descriptions title="相关信息" bordered>
                        <Descriptions.Item label="作者姓名" span={2}>叶其琛</Descriptions.Item>
                        <Descriptions.Item label="学校">北京大学</Descriptions.Item>
                        <Descriptions.Item label="项目介绍" span={2}>
                          此项目为作者的本科毕业设计项目
                        </Descriptions.Item>
                        <Descriptions.Item label="完成时间">
                          2021.05
                        </Descriptions.Item>
                        <Descriptions.Item label="仓库地址" span={2}>
                          https://github.com/yeeeqichen/KGQA
                        </Descriptions.Item>
                        <Descriptions.Item label="指导老师">
                          邹月娴 李素建
                        </Descriptions.Item>
                        <Descriptions.Item label="联系方式">
                          1700012775@pku.edu.cn
                        </Descriptions.Item>
                       
                      </Descriptions>
                    )
                  }
                })()
              }
            </Content>
          </Layout>
        </Layout>
      </Layout>
    )
  }
}
