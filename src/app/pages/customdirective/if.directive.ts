import { Directive,Input,TemplateRef,ViewContainerRef } from '@angular/core';

@Directive({
  selector: '[ngxIf]'
})
export class IfDirective {

  //1.What to Remove
  //2.From where to remove

  constructor(private template: TemplateRef<any>, private viewContainer: ViewContainerRef) {
  }

  @Input() set ngxIf(condition: boolean){
      if(condition){
        this.viewContainer.createEmbeddedView(this.template)
      } else {
      this.viewContainer.clear();
    }
  }
}
