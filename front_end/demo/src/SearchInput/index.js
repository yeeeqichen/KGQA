import React, { Component } from 'react'
import { Input } from 'antd';
import './index.css'

const { Search } = Input;

export default class SearchInput extends Component {
    render() {
        return (
            <div className="container">
                <Search
                    id="input"
                    placeholder="input search text"
                    allowClear
                    enterButton="Search"
                    size="large"
                    onSearch={this.props.onSearch}
                />
            </div>
            
        )
    }
}
