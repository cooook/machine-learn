pragma solidity ^0.4.23;
import "SafeMath.sol"

contract Caller {
    mapping(address => uint256) balances;

    function withdraw(uint256 _amount) {
        balances[msg.sender] = SafeMath.sub(balances[msg.sender], _amount);
    }
}
