import { Directive,ElementRef,OnInit,Renderer2 } from '@angular/core';

@Directive({
  selector: '[ngxHighlight]'
})
export class HighlightDirective implements OnInit {

  constructor(private element:ElementRef,private renderer:Renderer2) { }

  ngOnInit(){
    this.renderer.setStyle(this.element.nativeElement,'background-color','#F1948A');
    this.renderer.addClass(this.element.nativeElement,'container');
    this.renderer.setAttribute(this.element.nativeElement,'title','this is example div');
  }

}
