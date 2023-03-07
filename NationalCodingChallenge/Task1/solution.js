import cx from 'classnames';
import { Component } from 'react';

export default class Counter extends Component {
    state = {
    counter: 42
  };
    render() {
        return (
            <>
                <div className = "counter">
                    <h2>{this.state.counter}</h2>
                    </div>
                    <button class = "counter-button" onClick={() => this.setState({ counter: this.state.counter + 1})}>Click</button>;
                
                <style>{`
                    .counter-button {
                        font-size: 1rem;
                        padding: 5px 10px;
                        color:  #585858;
                    }
                `}</style>
            </>
        );
    }
}
