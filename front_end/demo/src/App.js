import React, { Component } from 'react'
import { Layout, Menu, Breadcrumb } from 'antd';
import Input from './SearchInput'
import Graph from './Graph'
import AnswerList from './AnswerList'
import { UserOutlined, LaptopOutlined, NotificationOutlined } from '@ant-design/icons';
import './App.css'
import $ from 'jquery'
const { SubMenu } = Menu;
const { Header, Content, Sider } = Layout;


export default class App extends Component {

  state = {answer: [], data: [], links: []}

  onSearch = (question) => {
    console.log(question)
    let _this = this
    $.ajax({
      url: 'http://localhost:2333/predict',
      data: {
        question:question
      },
      success: (res) => {
        let response_data = JSON.parse(res).data
        let { answers, sub_graph } = response_data
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
        console.log(data, links)
        _this.setState({
          answers: answers,
          data: data,
          links: links
        })
      }
    })
  }

  render() {
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
            >
              <SubMenu key="sub1" icon={<UserOutlined />} title="subnav 1">
                <Menu.Item key="1">option1</Menu.Item>
                <Menu.Item key="2">option2</Menu.Item>
                <Menu.Item key="3">option3</Menu.Item>
                <Menu.Item key="4">option4</Menu.Item>
              </SubMenu>
              <SubMenu key="sub2" icon={<LaptopOutlined />} title="subnav 2">
                <Menu.Item key="5">option5</Menu.Item>
                <Menu.Item key="6">option6</Menu.Item>
                <Menu.Item key="7">option7</Menu.Item>
                <Menu.Item key="8">option8</Menu.Item>
              </SubMenu>
              <SubMenu key="sub3" icon={<NotificationOutlined />} title="subnav 3">
                <Menu.Item key="9">option9</Menu.Item>
                <Menu.Item key="10">option10</Menu.Item>
                <Menu.Item key="11">option11</Menu.Item>
                <Menu.Item key="12">option12</Menu.Item>
              </SubMenu>
            </Menu>
          </Sider>
          <Layout style={{ padding: '0 24px 24px' }}>
            <Breadcrumb style={{ margin: '16px 0' }}>
              <Breadcrumb.Item>Home</Breadcrumb.Item>
              <Breadcrumb.Item>List</Breadcrumb.Item>
              <Breadcrumb.Item>App</Breadcrumb.Item>
            </Breadcrumb>
            <Content
              className="site-layout-background"
              style={{
                padding: 24,
                margin: 0,
                minHeight: 280,
              }}
            >
              <Input onSearch={this.onSearch}/>
              <div>
                <Graph data={this.state.data} links={this.state.links}/>
                <AnswerList/>
              </div>
            </Content>
          </Layout>
        </Layout>
      </Layout>
    )
  }
}
