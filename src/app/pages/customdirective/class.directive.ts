import { Directive, ElementRef, Renderer2 ,Input} from '@angular/core';

@Directive({
  selector: '[ngxClass]'
})
export class ClassDirective {

  constructor(private element: ElementRef, private renderer: Renderer2) { }

  @Input('ngxClass') set ngxClass(value:Object){
    let entries = Object.entries(value);

    for(let [className,condition] of entries){
      if(condition){
        this.renderer.addClass(this.element.nativeElement,className);
      }
    }
  }

}
