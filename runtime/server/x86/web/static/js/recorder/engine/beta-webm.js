/*
录音
https://github.com/xiangyuecn/Recorder
src: engine/beta-webm.js
*/
!function(){"use strict";var l="audio/webm",w=window.MediaRecorder&&MediaRecorder.isTypeSupported(l);Recorder.prototype.enc_webm={stable:!1,testmsg:w?"只有比较新的浏览器支持，压缩率和mp3差不多。由于未找到对已有pcm数据进行快速编码的方法，只能按照类似边播放边收听形式把数据导入到MediaRecorder，有几秒就要等几秒。(想接原始录音Stream？我不给，哼!)输出音频虽然可以通过比特率来控制文件大小，但音频文件中的比特率并非设定比特率，采样率由于是我们自己采样的，到这个编码器随他怎么搞":"此浏览器不支持进行webm编码，未实现MediaRecorder"},Recorder.prototype.webm=function(e,t,r){if(w){var a=this.set,n=e.length,o=a.sampleRate,c=Recorder.Ctx,i=c.createMediaStreamDestination();i.channelCount=1;var d=new MediaRecorder(i.stream,{mimeType:l,bitsPerSecond:1e3*a.bitRate}),s=[];d.ondataavailable=function(e){s.push(e.data)},d.onstop=function(e){t(new Blob(s,{type:l}))},d.onerror=function(e){r("转码webm出错："+e.message)},d.start();for(var m=c.createBuffer(1,n,o),u=m.getChannelData(0),p=0;p<n;p++){var f=e[p];f=f<0?f/32768:f/32767,u[p]=f}var b=c.createBufferSource();b.channelCount=1,b.buffer=m,b.connect(i),b.start?b.start():b.noteOn(0),b.onended=function(){d.stop()}}else r("此浏览器不支持把录音转成webm格式")}}();