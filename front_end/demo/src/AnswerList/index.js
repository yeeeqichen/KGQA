import React, { Component } from 'react'
import { List, Avatar, Space } from 'antd';
import './index.css'


export default class AnswerList extends Component {

    
    render() {
      const { answers } = this.props
      const dataList = answers.map((answer, index) => {
        return {
          title: answer,
          index: index,
          id: 'answer' + String(index + 1)
        }
      })
      return (
        <div className="list">
          <h3>预测答案及其对应子图</h3>
          <List
            itemLayout="vertical"
            size="default"
            pagination={{
              onChange: page => {
                console.log(page);
              },
              pageSize: 3,
            }}
            dataSource={dataList}
            renderItem={item => (
              <div className="panel panel-primary">
                <List.Item
                  key={item.title}
                  extra={
                    <div id={item.id}></div>
                  }
                  className='listItem'
                >
                  <List.Item.Meta
                    avatar={<Avatar className="avatar" src='/imgs/logo512.png' />}
                    title={<a href={item.href}>{item.title}</a>}
                    description={item.description}
                  />
                  {item.content}
                </List.Item>
                </div>
            )}
          />
          </div>
        )
    }
}
