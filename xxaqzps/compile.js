const solc = require('solc')
const fs = require('fs')
const shelljs = require('shelljs')
let args = process.argv.splice(2)
const sourcePath = args[0]
console.log(sourcePath)
// const sourcePath = '/home/chaoweilanmao/Desktop/tmp/'
const outputPath = args[2]
console.log(outputPath)
// let fileidx = parseInt(process.argv.splice(2)[0])
// let fileidx = args[1]
let filename = args[1]
let files = fs.readdirSync(sourcePath)

// The solc version after 0.5.0, we need a template to finish the compilation
// you can see the question in https://ethereum.stackexchange.com/questions/63267/solc-compile-0-5-0-is-not-working-like-0-4-version
let inputTemplate = 
{
    language: 'Solidity',
    sources: {
    },
    settings: {
        outputSelection: {
            "*": {
              "*" : [],
              "": ["legacyAST"]
            }
          }
    }
};

// console.log(fileidx)
// let codePath = sourcePath + files[fileidx]
let codePath = sourcePath + filename

console.log(codePath)

let codeContent = fs.readFileSync(codePath, 'UTF-8').toString()//读取文件内容

let version = solc.version().toString().split('+')[0].split('.')[1]//获取版本
console.log(version!=='4')
let idx = 0
// discriminate the version of solc
if(version !== '4') {
  inputTemplate.sources[idx] = {}
  inputTemplate.sources[idx]['content'] = codeContent
  // console.log(inputTemplate.sources[idx]['content'])
  try {
    // console.log(1)
    // console.log(inputTemplate)
    let output = JSON.parse(solc.compile(JSON.stringify(inputTemplate)));
    console.log(1)
    console.log(JSON.stringify(inputTemplate))
    // console.log(JSON.stringify(output['sources'][files[filename]]['legacyAST']))
    fs.writeFileSync(outputPath+filename.split('.')[0]+'.ast', JSON.stringify(output['sources'][files[filename]]['legacyAST']))
    
  } catch(e) {
    console.log(e.toString())
  }
} else {
  console.log(1)
  try {
    console.log(codeContent)
    let output = solc.compile(codeContent, (res)=>{})
    console.log(2)
    fs.writeFileSync(outputPath+filename.split('.')[0]+'.ast', JSON.stringify(output['sources']['']['AST']))
    // console.log(JSON.stringify(output['sources'][files[filename]]['legacyAST']))
  } catch(e) {
    console.log(e.toString())
  }
}


// node compile.js 源目录/ xxx.sol 目标目录/