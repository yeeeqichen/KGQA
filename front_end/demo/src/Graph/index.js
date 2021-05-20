import React, { Component } from 'react'
import * as echarts from 'echarts'
import './index.css'

export default class Graph extends Component {

    state = { displayGraph: true }
    
    componentDidUpdate() {
        this.componentDidMount()
    }
    
    componentDidMount() {
        let chartDom = document.getElementById('graph');
        let myChart = echarts.init(chartDom);
        let option;
        option = {
            title: {
                text: '关系子图',
                subtext: 'Default layout',
            },
            tooltip: {},
            animationDuration: 1500,
            animationEasingUpdate: 'quinticInOut',
            series: [
                {
                    name: '关系子图',
                    type: 'graph',
                    layout: 'force',
                    data: this.props.data,
                    links: this.props.links,
                    roam: 'move',
                    zoom: 5.5,
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
    render() {
        return (
            <div id="graph" className={ this.state.displayGraph? '_': 'notdisplay'}>
                
            </div>
        )
    }
}
