const katex=require('katex'),fs=require('fs');
const items=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));let bad=0;
const out=items.map(([tex,display])=>{
  try{return katex.renderToString(tex,{displayMode:display,output:'html',throwOnError:true,strict:'ignore'});}
  catch(e){bad++;console.error('KATEX ERR',e.message.slice(0,120),'|',tex.slice(0,80));
    return katex.renderToString(tex,{displayMode:display,output:'html',throwOnError:false,strict:'ignore'});}
});
fs.writeFileSync(process.argv[3],JSON.stringify(out));console.error('math items',items.length,'errors',bad);
